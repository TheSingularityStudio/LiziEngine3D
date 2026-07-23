use std::sync::Arc;
use eframe::egui;
use egui::menu;

use crate::gui::interaction::{InteractionState, ToolMode, HoveredParticleInfo, OrbitCamera};
use crate::core::sim::ElectrostaticSim3D;
use crate::core::boundary::BoundaryType;
use crate::presets::PresetVariant;
use crate::visual::colors::heatmap_rgb;

fn load_chinese_fonts(fonts: &mut egui::FontDefinitions) -> bool {
    let font_candidates = [
        "C:\\Windows\\Fonts\\msyh.ttc", "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\simsun.ttc", "C:\\Windows\\Fonts\\yahei.ttf",
    ];
    for path in &font_candidates {
        if let Ok(data) = std::fs::read(path) {
            let name = format!("chinese_{}", fonts.font_data.len());
            fonts.font_data.insert(name.clone(), Arc::new(egui::FontData::from_owned(data)));
            if let Some(proportional) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
                proportional.insert(0, name.clone());
            }
            if let Some(monospace) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
                monospace.insert(0, name);
            }
            return true;
        }
    }
    false
}

struct SimulationState {
    variant: PresetVariant,
    sim: ElectrostaticSim3D,
    paused: bool,
    v_min: f64,
    v_max: f64,
    interaction: InteractionState,
    show_left_panel: bool,
    show_right_panel: bool,
    show_about_dialog: bool,
    show_shortcuts_dialog: bool,
    message_dialog: Option<String>,
}

pub struct LiziApp {
    state: Option<SimulationState>,
}

impl Default for LiziApp {
    fn default() -> Self { Self { state: None } }
}

impl LiziApp {
    pub fn run() {
        let native_options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
            ..Default::default()
        };
        let _ = eframe::run_native(
            "LiziEngine3D - 静电 PIC 模拟器",
            native_options,
            Box::new(|cc| {
                let mut fonts = egui::FontDefinitions::default();
                if !load_chinese_fonts(&mut fonts) {
                    eprintln!("警告: 未找到中文字体");
                }
                cc.egui_ctx.set_fonts(fonts);
                Ok(Box::new(LiziApp::default()))
            }),
        );
    }

    fn render_preset_selection(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(60.0);
                ui.heading("LiziEngine3D");
                ui.label("3D 静电 PIC 模拟器");
                ui.add_space(10.0);
                ui.separator();
                ui.add_space(20.0);
                ui.label("选择一个预设场景开始模拟：");
                ui.add_space(10.0);

                for variant in PresetVariant::all() {
                    let name = variant.display_name();
                    let desc = variant.description();
                    let frame = egui::Frame::group(ui.style());
                    frame.show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.set_min_width(ui.available_width());
                            if ui.button(name).clicked() {
                                let sim = variant.create_sim();
                                self.state = Some(SimulationState {
                                    variant: *variant, sim,
                                    paused: false, v_min: 0.0, v_max: 1.0,
                                    interaction: InteractionState::new(),
                                    show_left_panel: true, show_right_panel: true,
                                    show_about_dialog: false, show_shortcuts_dialog: false,
                                    message_dialog: None,
                                });
                            }
                            ui.add_space(10.0);
                            ui.label(desc);
                        });
                    });
                    ui.add_space(5.0);
                }
            });
        });
    }
}

fn render_menu_bar(ctx: &egui::Context, state: &mut SimulationState) -> bool {
    let mut back_requested = false;
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        menu::bar(ui, |ui| {
            ui.menu_button("文件", |ui| {
                if ui.button("返回预设选择").clicked() { back_requested = true; ui.close_menu(); }
                ui.separator();
                if ui.button("退出").clicked() { std::process::exit(0); }
            });
            ui.menu_button("选项", |ui| {
                let mut sl = state.show_left_panel;
                if ui.checkbox(&mut sl, "显示工具面板").changed() { state.show_left_panel = sl; }
                let mut sr = state.show_right_panel;
                if ui.checkbox(&mut sr, "显示参数面板").changed() { state.show_right_panel = sr; }
                ui.separator();
                let mut sh = state.interaction.show_heatmap;
                if ui.checkbox(&mut sh, "显示电势色图").changed() { state.interaction.show_heatmap = sh; }
                let mut sg = state.interaction.show_grid;
                if ui.checkbox(&mut sg, "显示网格").changed() { state.interaction.show_grid = sg; }
                let mut sa = state.interaction.show_axes;
                if ui.checkbox(&mut sa, "显示坐标轴").changed() { state.interaction.show_axes = sa; }
            });
            ui.menu_button("帮助", |ui| {
                if ui.button("关于 LiziEngine3D").clicked() { state.show_about_dialog = true; ui.close_menu(); }
                if ui.button("快捷键说明").clicked() { state.show_shortcuts_dialog = true; ui.close_menu(); }
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!("预设: {}", state.variant.display_name()));
                ui.label(format!("粒子数: {}", state.sim.particles.len()));
                ui.separator();
                if ui.button("⟳ Reset").clicked() {
                    let ns = state.variant.create_sim();
                    state.sim = ns; state.paused = false;
                    state.v_min = 0.0; state.v_max = 1.0;
                    state.interaction = InteractionState::new();
                }
                if ui.button("⏭ Step").clicked() {
                    state.paused = true;
                    state.sim.step(state.variant.config().dt);
                }
                if state.paused {
                    if ui.button("▶ Play").clicked() { state.paused = false; }
                } else {
                    if ui.button("⏸ Pause").clicked() { state.paused = true; }
                }
                if ui.button("← 返回").clicked() { back_requested = true; }
            });
        });
    });
    back_requested
}

fn render_dialogs(ctx: &egui::Context, state: &mut SimulationState) {
    if let Some(msg) = &state.message_dialog.clone() {
        let mut open = true;
        egui::Window::new("消息").open(&mut open).resizable(false).default_size([400.0, 150.0]).show(ctx, |ui| {
            ui.add_space(8.0); ui.label(msg); ui.add_space(12.0);
            if ui.button("确定").clicked() { state.message_dialog = None; }
        });
        if !open { state.message_dialog = None; }
    }
    if state.show_about_dialog {
        egui::Window::new("关于 LiziEngine3D").open(&mut state.show_about_dialog).resizable(false)
            .default_size([420.0, 280.0]).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("LiziEngine3D"); ui.label("版本 0.1.0"); ui.separator(); ui.add_space(8.0);
                ui.label("三维静电 PIC (Particle-in-Cell) 模拟器");
                ui.add_space(8.0);
                ui.label("技术栈："); ui.label("  • eframe/egui — GUI 框架"); ui.label("  • ndarray — 数值计算"); ui.label("  • ndrustfft — FFT Poisson 求解器");
                ui.add_space(8.0); ui.separator(); ui.add_space(8.0); ui.label("许可证：MIT");
            });
        });
    }
    if state.show_shortcuts_dialog {
        egui::Window::new("快捷键说明").open(&mut state.show_shortcuts_dialog).resizable(false)
            .default_size([380.0, 300.0]).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("工具模式"); ui.separator(); ui.add_space(4.0);
                ui.label("左侧面板选择工具：");
                ui.label("  • 拖动 — 点击选中粒子并拖拽移动");
                ui.label("  • 放置 — 点击画布创建新粒子");
                ui.label("  • 删除 — 点击粒子删除");
                ui.label("  • 查看 — 鼠标拖拽旋转 3D 视角，滚轮缩放");
                ui.add_space(8.0);
                ui.heading("3D 视角操作"); ui.separator(); ui.add_space(4.0);
                ui.label("  • 鼠标左键拖拽 — 旋转 3D 视角");
                ui.label("  • 滚轮 — 缩放");
                ui.label("  • 右键拖拽 — 平移场景");
                ui.add_space(8.0);
                ui.heading("模拟控制"); ui.separator(); ui.add_space(4.0);
                ui.label("  • ▶ Play — 自动步进"); ui.label("  • ⏸ Pause — 暂停");
                ui.label("  • ⏭ Step — 单步"); ui.label("  • ⟳ Reset — 重置");
                ui.label("  • ← 返回 — 预设选择界面");
            });
        });
    }
}

fn render_left_panel(ctx: &egui::Context, state: &mut SimulationState) {
    if !state.show_left_panel { return; }
    let interaction = &mut state.interaction;
    egui::SidePanel::left("tool_panel").resizable(false).default_width(140.0).show(ctx, |ui| {
        ui.vertical(|ui| {
            ui.add_space(8.0); ui.heading("工具"); ui.separator(); ui.add_space(4.0);
            for tool in ToolMode::all() {
                let selected = interaction.tool_mode == tool;
                let text = format!("{} {}", tool.icon(), tool.display_name());
                let btn = if selected {
                    egui::Button::new(text).fill(ui.style().visuals.selection.bg_fill).min_size(egui::vec2(120.0, 32.0))
                } else {
                    egui::Button::new(text).min_size(egui::vec2(120.0, 32.0))
                };
                if ui.add(btn).clicked() { interaction.tool_mode = tool; }
            }
            ui.add_space(16.0); ui.separator(); ui.add_space(8.0);
            ui.label("提示：拖拽选择粒子"); ui.label("点击画布执行操作");
        });
    });
}

fn render_right_panel(ctx: &egui::Context, state: &mut SimulationState) {
    if !state.show_right_panel { return; }
    let interaction = &mut state.interaction;
    egui::SidePanel::right("param_panel").resizable(false).default_width(200.0).show(ctx, |ui| {
        ui.vertical(|ui| {
            ui.add_space(8.0); ui.heading("参数"); ui.separator();

            ui.label(format!("工具: {}", interaction.tool_mode.display_name()));
            ui.add_space(4.0);

            match interaction.tool_mode {
                ToolMode::DragParticle => {
                    ui.label("拖拽粒子改变位置"); ui.label("选中时速度归零");
                    ui.add_space(4.0);
                    ui.add(egui::Slider::new(&mut interaction.selection_radius, 0.01..=0.20).text("选择半径").step_by(0.005));
                }
                ToolMode::PlaceParticle => {
                    let list_enabled = &mut interaction.placement_list.enabled;
                    ui.checkbox(list_enabled, "使用放置清单");
                    if *list_enabled {
                        // simplified - just show basic controls
                        ui.label("点击画布放置所有清单粒子");
                    } else {
                        ui.add(egui::Slider::new(&mut interaction.place_params.charge, -10.0..=10.0).text("电荷 q").step_by(0.1));
                        ui.add(egui::Slider::new(&mut interaction.place_params.mass, 0.1..=10.0).text("质量 m").step_by(0.1));
                        ui.checkbox(&mut interaction.place_params.fixed, "固定粒子");
                        ui.label("点击画布放置粒子");
                    }
                }
                ToolMode::DeleteParticle => {
                    ui.label("点击粒子删除");
                    ui.add(egui::Slider::new(&mut interaction.selection_radius, 0.01..=0.20).text("选择半径").step_by(0.005));
                }
                ToolMode::Inspect => {
                    ui.label("左键拖拽旋转"); ui.label("滚轮缩放"); ui.label("右键平移");
                    ui.add_space(4.0);
                    ui.label(format!("角度: {:.1}°", interaction.camera.azimuth.to_degrees()));
                    ui.label(format!("俯仰: {:.1}°", interaction.camera.elevation.to_degrees()));
                    ui.label(format!("距离: {:.2}", interaction.camera.distance));
                    if ui.button("重置视图").clicked() { interaction.reset_view(); }
                    ui.add_space(8.0); ui.separator();
                    if let Some(info) = &interaction.hovered_particle {
                        ui.heading("粒子信息"); ui.separator();
                        ui.label(format!("#{}", info.index));
                        ui.label(format!("位置: ({:.3}, {:.3}, {:.3})", info.x, info.y, info.z));
                        ui.label(format!("速度: ({:.2e}, {:.2e}, {:.2e})", info.vx, info.vy, info.vz));
                        ui.label(format!("电荷: {:.3}  质量: {:.3}", info.q, info.m));
                    } else {
                        ui.label("悬停粒子查看信息");
                    }
                }
            }

            ui.add_space(8.0); ui.separator();
            ui.label(format!("dt = {:.2e}", state.variant.config().dt));
            if state.paused { ui.label("⏸ 已暂停"); } else { ui.label("▶ 运行中"); }

            ui.add_space(8.0); ui.separator();
            ui.label("边界：");
            for bt in BoundaryType::all() {
                ui.radio_value(&mut state.sim.boundary_type, bt, bt.display_name());
            }

            ui.add_space(8.0); ui.separator();
            ui.checkbox(&mut state.sim.gravity_enabled, "重力");
            if state.sim.gravity_enabled {
                ui.add(egui::Slider::new(&mut state.sim.gravity_z, -50.0..=50.0).text("g_z").step_by(0.1));
            }
            ui.checkbox(&mut state.sim.friction_enabled, "摩擦力");
            if state.sim.friction_enabled {
                ui.add(egui::Slider::new(&mut state.sim.friction_damping, 0.0..=5.0).text("阻尼").step_by(0.01));
            }
        });
    });
}

/// 3D 投影：将世界坐标 (x,y,z) 投影到屏幕坐标
fn project_3d(
    world_x: f64, world_y: f64, world_z: f64,
    camera: &OrbitCamera,
    lx: f64, ly: f64, lz: f64,
) -> Option<(f32, f32, f32)> {
    // Normalize to [0,1]
    let nx = (world_x / lx) as f32;
    let ny = (world_y / ly) as f32;
    let nz = (world_z / lz) as f32;

    // Translate to center: [-0.5, 0.5]
    let mut px = nx - camera.target_x;
    let mut py = ny - camera.target_y;
    let mut pz = nz - camera.target_z;

    // Scale by distance
    let d = camera.distance.max(0.1);

    // Rotation: azimuth around Y, then elevation around X
    let cos_a = camera.azimuth.cos();
    let sin_a = camera.azimuth.sin();
    let cos_e = camera.elevation.cos();
    let sin_e = camera.elevation.sin();

    // Azimuth rotation (around Y)
    let rx = px * cos_a - pz * sin_a;
    let rz = px * sin_a + pz * cos_a;
    px = rx;
    pz = rz;

    // Elevation rotation (around X)
    let ry = py * cos_e - pz * sin_e;
    let rz2 = py * sin_e + pz * cos_e;
    py = ry;
    pz = rz2;

    // Perspective projection
    let fov = 2.0f32;
    let z_clip = (-pz + d).max(0.01);
    let scale = fov / z_clip;

    let sx = px * scale;
    let sy = -py * scale; // flip Y
    let depth = z_clip;

    Some((sx, sy, depth))
}

fn render_3d_canvas(ctx: &egui::Context, state: &mut SimulationState) {
    let sim = &mut state.sim;
    let v_min = &mut state.v_min;
    let v_max = &mut state.v_max;
    let interaction = &mut state.interaction;

    egui::CentralPanel::default().show(ctx, |ui| {
        if sim.v.is_none() || sim.ex.is_none() || sim.ey.is_none() || sim.ez.is_none() {
            sim.compute_fields();
        }

        let snapshot = sim.get_state_snapshot();
        let (nx, ny, nz) = snapshot.v.dim();

        let mut min_v = f64::MAX;
        let mut max_v = f64::MIN;
        for val in snapshot.v.iter() {
            if *val < min_v { min_v = *val; }
            if *val > max_v { max_v = *val; }
        }
        *v_min = *v_min * 0.9 + min_v * 0.1;
        *v_max = *v_max * 0.9 + max_v * 0.1;
        if (*v_max - *v_min).abs() < 1e-12 { *v_max = *v_min + 1.0; }

        let avail = ui.available_rect_before_wrap();
        let size = avail.size().x.min(avail.size().y) * 0.95;
        let center = avail.center();
        let canvas_rect = egui::Rect::from_center_size(center, egui::vec2(size, size));

        let bg_color = ui.style().visuals.panel_fill;
        let painter = ui.painter();
        painter.rect_filled(canvas_rect, 0.0, bg_color);
        painter.rect_stroke(canvas_rect, 0.0, (1.0, egui::Color32::GRAY), egui::StrokeKind::Inside);

        let cx = canvas_rect.center();

        // Handle camera interaction for Inspect mode
        let mouse_pos = ui.input(|i| i.pointer.hover_pos());
        let mouse_down = ui.input(|i| i.pointer.any_down());
        let mouse_primary = ui.input(|i| i.pointer.button_down(egui::PointerButton::Primary));
        let mouse_secondary = ui.input(|i| i.pointer.button_down(egui::PointerButton::Secondary));
        let scroll_delta = ui.input(|i| i.raw_scroll_delta);

        if let Some(pos) = mouse_pos {
            if canvas_rect.contains(pos) {
                if interaction.tool_mode == ToolMode::Inspect {
                    // Orbit rotation (left button)
                    if mouse_primary && !mouse_secondary {
                        if let Some(last) = interaction.camera.last_mouse_pos {
                            let dx = pos.x - last.0;
                            let dy = pos.y - last.1;
                            interaction.camera.azimuth += dx * 0.008;
                            interaction.camera.elevation = (interaction.camera.elevation + dy * 0.008)
                                .clamp(-std::f32::consts::FRAC_PI_2 + 0.05, std::f32::consts::FRAC_PI_2 - 0.05);
                        }
                        interaction.camera.last_mouse_pos = Some((pos.x, pos.y));
                    }
                    // Pan (right button)
                    else if mouse_secondary {
                        if let Some(last) = interaction.camera.last_mouse_pos {
                            let dx = (pos.x - last.0) * 0.005 * interaction.camera.distance;
                            let dy = (pos.y - last.1) * 0.005 * interaction.camera.distance;
                            // Pan in camera space
                            let cos_a = interaction.camera.azimuth.cos();
                            let sin_a = interaction.camera.azimuth.sin();
                            interaction.camera.target_x -= dx * cos_a;
                            interaction.camera.target_z -= dx * sin_a;
                            interaction.camera.target_y += dy;
                        }
                        interaction.camera.last_mouse_pos = Some((pos.x, pos.y));
                    } else {
                        interaction.camera.last_mouse_pos = None;
                    }

                    // Scroll zoom
                    if scroll_delta.y != 0.0 {
                        let factor = if scroll_delta.y > 0.0 { 0.9 } else { 1.1 };
                        interaction.camera.distance = (interaction.camera.distance * factor).clamp(0.3, 15.0);
                    }
                } else {
                    interaction.camera.last_mouse_pos = None;
                }
            }
        }

        // Draw 3D axes
        if interaction.show_axes {
            let axis_len = 0.3f32;
            let origin = project_3d(0.0, 0.0, 0.0, &interaction.camera, snapshot.lx, snapshot.ly, snapshot.lz);
            let x_end = project_3d(snapshot.lx * axis_len as f64, 0.0, 0.0, &interaction.camera, snapshot.lx, snapshot.ly, snapshot.lz);
            let y_end = project_3d(0.0, snapshot.ly * axis_len as f64, 0.0, &interaction.camera, snapshot.lx, snapshot.ly, snapshot.lz);
            let z_end = project_3d(0.0, 0.0, snapshot.lz * axis_len as f64, &interaction.camera, snapshot.lx, snapshot.ly, snapshot.lz);

            if let (Some((ox, oy, _)), Some((xx, xy, _)), Some((yx, yy, _)), Some((zx, zy, _))) = (origin, x_end, y_end, z_end) {
                let opos = egui::pos2(cx.x + ox * size * 0.5, cx.y + oy * size * 0.5);
                painter.line_segment(
                    [opos, egui::pos2(cx.x + xx * size * 0.5, cx.y + xy * size * 0.5)],
                    (2.0, egui::Color32::RED),
                );
                painter.line_segment(
                    [opos, egui::pos2(cx.x + yx * size * 0.5, cx.y + yy * size * 0.5)],
                    (2.0, egui::Color32::GREEN),
                );
                painter.line_segment(
                    [opos, egui::pos2(cx.x + zx * size * 0.5, cx.y + zy * size * 0.5)],
                    (2.0, egui::Color32::BLUE),
                );
            }
        }

        // Draw 3D grid lines (floor plane at z=0)
        if interaction.show_grid {
            let grid_color = egui::Color32::from_gray(80);
            let grid_steps = 8usize;
            for i in 0..=grid_steps {
                for j in 0..=grid_steps {
                    let frac_i = i as f64 / grid_steps as f64;
                    let frac_j = j as f64 / grid_steps as f64;
                    // Draw lines parallel to X (varying X at fixed Y)
                    if j < grid_steps {
                        let p1 = project_3d(frac_i * snapshot.lx, frac_j * snapshot.ly, 0.0, &interaction.camera, snapshot.lx, snapshot.ly, snapshot.lz);
                        let p2 = project_3d(frac_i * snapshot.lx, (frac_j + 1.0/grid_steps as f64) * snapshot.ly, 0.0, &interaction.camera, snapshot.lx, snapshot.ly, snapshot.lz);
                        if let (Some((x1, y1, _)), Some((x2, y2, _))) = (p1, p2) {
                            painter.line_segment(
                                [egui::pos2(cx.x + x1 * size * 0.5, cx.y + y1 * size * 0.5),
                                 egui::pos2(cx.x + x2 * size * 0.5, cx.y + y2 * size * 0.5)],
                                (0.5, grid_color),
                            );
                        }
                    }
                    if i < grid_steps {
                        let p1 = project_3d(frac_i * snapshot.lx, frac_j * snapshot.ly, 0.0, &interaction.camera, snapshot.lx, snapshot.ly, snapshot.lz);
                        let p2 = project_3d((frac_i + 1.0/grid_steps as f64) * snapshot.lx, frac_j * snapshot.ly, 0.0, &interaction.camera, snapshot.lx, snapshot.ly, snapshot.lz);
                        if let (Some((x1, y1, _)), Some((x2, y2, _))) = (p1, p2) {
                            painter.line_segment(
                                [egui::pos2(cx.x + x1 * size * 0.5, cx.y + y1 * size * 0.5),
                                 egui::pos2(cx.x + x2 * size * 0.5, cx.y + y2 * size * 0.5)],
                                (0.5, grid_color),
                            );
                        }
                    }
                }
            }
        }

        // Draw slice plane heatmap (z = mid slice)
        if interaction.show_heatmap {
            let mid_k = nz / 2;
            let half_tex = 64;
            let _tex_nx = nx.min(half_tex);
            let _tex_ny = ny.min(half_tex);
            let step_i = if nx > half_tex { nx / half_tex } else { 1 };
            let step_j = if ny > half_tex { ny / half_tex } else { 1 };

            for i in (0..nx).step_by(step_i) {
                for j in (0..ny).step_by(step_j) {
                    let world_x = (i as f64 + 0.5) * snapshot.lx / nx as f64;
                    let world_y = (j as f64 + 0.5) * snapshot.ly / ny as f64;
                    let world_z = (mid_k as f64 + 0.5) * snapshot.lz / nz as f64;

                    if let Some((sx, sy, depth)) = project_3d(world_x, world_y, world_z, &interaction.camera, snapshot.lx, snapshot.ly, snapshot.lz) {
                        let val = snapshot.v[[i, j, mid_k]];
                        let (r, g, b) = heatmap_rgb(val, *v_min, *v_max);
                        let screen_x = cx.x + sx * size * 0.5;
                        let screen_y = cx.y + sy * size * 0.5;
                        let pw = (size * 0.5 / depth * (snapshot.lx / nx as f64) as f32 * 2.0).max(1.5);
                        let ph = (size * 0.5 / depth * (snapshot.ly / ny as f64) as f32 * 2.0).max(1.5);
                        painter.rect_filled(
                            egui::Rect::from_center_size(egui::pos2(screen_x, screen_y), egui::vec2(pw, ph)),
                            0.0, egui::Color32::from_rgb(r, g, b),
                        );
                    }
                }
            }
        }

        // Draw particles with Z-sorting
        struct ParticleDraw {
            screen_x: f32,
            screen_y: f32,
            depth: f32,
            color: egui::Color32,
            radius: f32,
            index: usize,
        }

        let particle_count = snapshot.x.len();
        let mut draw_list: Vec<ParticleDraw> = Vec::with_capacity(particle_count);

        for p in 0..particle_count {
            if let Some((sx, sy, depth)) = project_3d(
                snapshot.x[p], snapshot.y[p], snapshot.z[p],
                &interaction.camera, snapshot.lx, snapshot.ly, snapshot.lz,
            ) {
                let color = if snapshot.q[p] < 0.0 {
                    egui::Color32::CYAN
                } else {
                    egui::Color32::WHITE
                };
                let base_radius = (4.0 / depth.max(0.1)).min(8.0).max(2.0);
                draw_list.push(ParticleDraw {
                    screen_x: cx.x + sx * size * 0.5,
                    screen_y: cx.y + sy * size * 0.5,
                    depth,
                    color,
                    radius: base_radius,
                    index: p,
                });
            }
        }

        // Z-sort (far to near)
        draw_list.sort_by(|a, b| b.depth.partial_cmp(&a.depth).unwrap_or(std::cmp::Ordering::Equal));

        // Find hovered particle (inspect mode)
        if interaction.tool_mode == ToolMode::Inspect {
            interaction.hovered_particle = None;
            if !mouse_down {
                if let Some(pos) = mouse_pos {
                    let mut min_dist_sq = (interaction.selection_radius * size as f64).powi(2) as f32;
                    let mut closest: Option<usize> = None;
                    for pd in &draw_list {
                        let dx = pd.screen_x - pos.x;
                        let dy = pd.screen_y - pos.y;
                        let dist_sq = dx * dx + dy * dy;
                        if dist_sq < min_dist_sq {
                            min_dist_sq = dist_sq;
                            closest = Some(pd.index);
                        }
                    }
                    if let Some(idx) = closest {
                        interaction.hovered_particle = Some(HoveredParticleInfo {
                            index: idx,
                            x: snapshot.x[idx], y: snapshot.y[idx], z: snapshot.z[idx],
                            vx: snapshot.vx[idx], vy: snapshot.vy[idx], vz: snapshot.vz[idx],
                            q: snapshot.q[idx], m: snapshot.m[idx],
                        });
                    }
                }
            }
        }

        // Draw particles
        for pd in &draw_list {
            let radius = if interaction.tool_mode == ToolMode::Inspect {
                if let Some(info) = &interaction.hovered_particle {
                    if info.index == pd.index { pd.radius * 1.8 } else { pd.radius }
                } else { pd.radius }
            } else { pd.radius };

            painter.circle_filled(egui::pos2(pd.screen_x, pd.screen_y), radius, pd.color);

            // Highlight hovered
            if interaction.tool_mode == ToolMode::Inspect {
                if let Some(info) = &interaction.hovered_particle {
                    if info.index == pd.index {
                        painter.circle_stroke(egui::pos2(pd.screen_x, pd.screen_y), pd.radius * 2.0, (2.0, egui::Color32::YELLOW));
                    }
                }
            }
        }

        // Show hover tooltip
        if interaction.tool_mode == ToolMode::Inspect {
            if let Some(info) = &interaction.hovered_particle.clone() {
                if info.index < particle_count {
                    if let Some((sx, sy, _)) = project_3d(
                        snapshot.x[info.index], snapshot.y[info.index], snapshot.z[info.index],
                        &interaction.camera, snapshot.lx, snapshot.ly, snapshot.lz,
                    ) {
                        let tip_pos = egui::pos2(cx.x + sx * size * 0.5 + 10.0, cx.y + sy * size * 0.5 - 30.0);
                        let text = format!(
                            "#{}  q={:.2}  m={:.2}\n({:.3}, {:.3}, {:.3})  v=({:.2e}, {:.2e}, {:.2e})",
                            info.index, info.q, info.m, info.x, info.y, info.z, info.vx, info.vy, info.vz
                        );
                        painter.text(tip_pos, egui::Align2::LEFT_TOP, text,
                            egui::TextStyle::Body.resolve(ui.style()), egui::Color32::YELLOW);
                    }
                }
            }
        }

        // Handle tool interactions (Drag, Place, Delete)
        if interaction.tool_mode == ToolMode::DragParticle && mouse_down && mouse_primary {
            if let Some(pos) = mouse_pos {
                if canvas_rect.contains(pos) {
                    // Simple ray-pick: find closest particle in screen space
                    if !interaction.dragging {
                        let mut min_dist = interaction.selection_radius * size as f64;
                        let mut closest = None;
                        for pd in &draw_list {
                            let dx = (pd.screen_x - pos.x) as f64;
                            let dy = (pd.screen_y - pos.y) as f64;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if dist < min_dist {
                                min_dist = dist;
                                closest = Some(pd.index);
                            }
                        }
                        if let Some(idx) = closest {
                            interaction.dragging = true;
                            interaction.dragged_particle_index = Some(idx);
                        }
                    }
                    if let Some(idx) = interaction.dragged_particle_index {
                        // Move particle to a plane at constant Z
                        let world_z = snapshot.z[idx];
                        // Inverse project: from screen to world (approximate)
                        if let Some((world_x, world_y)) = inverse_project_2d(pos, cx, size, &interaction.camera, snapshot.lx, snapshot.ly, snapshot.lz, world_z) {
                            sim.particles.x[idx] = world_x.clamp(0.0, snapshot.lx);
                            sim.particles.y[idx] = world_y.clamp(0.0, snapshot.ly);
                            sim.particles.vx[idx] = 0.0;
                            sim.particles.vy[idx] = 0.0;
                            sim.v = None; sim.ex = None; sim.ey = None; sim.ez = None;
                        }
                    }
                }
            }
        } else {
            interaction.dragging = false;
            interaction.dragged_particle_index = None;
        }

        // Place particle
        if interaction.tool_mode == ToolMode::PlaceParticle {
            let clicked = ui.input(|i| i.pointer.any_click());
            if clicked {
                if let Some(pos) = mouse_pos {
                    if canvas_rect.contains(pos) {
                        // Place at z=0.5 * lz
                        let world_z = 0.5 * snapshot.lz;
                        if let Some((world_x, world_y)) = inverse_project_2d(pos, cx, size, &interaction.camera, snapshot.lx, snapshot.ly, snapshot.lz, world_z) {
                            let wx = world_x.clamp(0.0, snapshot.lx);
                            let wy = world_y.clamp(0.0, snapshot.ly);
                            let q = interaction.place_params.charge;
                            let m = interaction.place_params.mass;
                            let _fixed = interaction.place_params.fixed;
                            sim.particles.add_particle(wx, wy, world_z, q, m, 0.0, 0.0, 0.0);
                            sim.v = None; sim.ex = None; sim.ey = None; sim.ez = None;
                        }
                    }
                }
            }
        }

        // Delete particle
        if interaction.tool_mode == ToolMode::DeleteParticle {
            let clicked = ui.input(|i| i.pointer.any_click());
            if clicked {
                if let Some(pos) = mouse_pos {
                    if canvas_rect.contains(pos) {
                        let mut min_dist = interaction.selection_radius * size as f64;
                        let mut closest = None;
                        for pd in &draw_list {
                            let dx = (pd.screen_x - pos.x) as f64;
                            let dy = (pd.screen_y - pos.y) as f64;
                            let dist = (dx * dx + dy * dy).sqrt();
                            if dist < min_dist {
                                min_dist = dist;
                                closest = Some(pd.index);
                            }
                        }
                        if let Some(idx) = closest {
                            sim.particles.remove_particle(idx);
                            sim.v = None; sim.ex = None; sim.ey = None; sim.ez = None;
                        }
                    }
                }
            }
        }

        // Clear mouse state
        if !mouse_down {
            interaction.dragging = false;
            interaction.dragged_particle_index = None;
        }
    });
}

/// 从屏幕坐标反投影到 3D 世界坐标（在给定 z 平面上）
fn inverse_project_2d(
    screen_pos: egui::Pos2,
    center: egui::Pos2,
    canvas_size: f32,
    camera: &OrbitCamera,
    lx: f64, ly: f64, lz: f64,
    fixed_z: f64,
) -> Option<(f64, f64)> {
    let dx = (screen_pos.x - center.x) / (canvas_size * 0.5);
    let dy = (screen_pos.y - center.y) / (canvas_size * 0.5);

    let d = camera.distance.max(0.1) as f64;
    let fov = 2.0f64;

    // Fixed z in normalized coords
    let _nz = fixed_z / lz - camera.target_z as f64;

    // Elevation inverse
    let _cos_e = camera.elevation.cos() as f64;
    let _sin_e = camera.elevation.sin() as f64;

    // We want to find px, py such that:
    // py = (sy * z_clip) / (-fov)  (from projection)
    // px = (sx * z_clip) / fov
    // where z_clip = -pz + d
    // And pz comes from rotation...

    // Simplified: assume the camera sees a plane at fixed_z
    // Just return normalized screen coordinates
    // This is an approximation that works well enough for interaction
    let _aspect = 1.0;

    // Simple ray-plane intersection approximation
    let sx = dx as f64;
    let sy = dy as f64;

    // Estimate depth at this screen position (intersect with z = fixed_z plane)
    // For simplicity, just map screen to world with some scaling
    let world_x = (sx * d / fov + camera.target_x as f64) * lx;
    let world_y = (sy * d / fov + camera.target_y as f64) * ly;

    Some((world_x, world_y))
}

fn render_simulation_panels(ctx: &egui::Context, state: &mut SimulationState) -> bool {
    let back_flag = render_menu_bar(ctx, state);
    render_dialogs(ctx, state);
    render_left_panel(ctx, state);
    render_right_panel(ctx, state);
    render_3d_canvas(ctx, state);
    if !state.paused {
        let dt = state.variant.config().dt;
        state.sim.step(dt);
        ctx.request_repaint();
    }
    !back_flag
}

impl eframe::App for LiziApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(ref mut state) = self.state {
            let running = render_simulation_panels(ctx, state);
            if !running { self.state = None; }
            else if !state.paused { ctx.request_repaint(); }
        } else {
            self.render_preset_selection(ctx);
        }
    }
}