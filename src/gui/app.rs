use std::sync::Arc;
use eframe::egui;
use egui::ColorImage;
use egui::TextureHandle;
use egui::load::SizedTexture;
use egui::menu;

use crate::gui::interaction::{InteractionState, ToolMode, ArrangeMode, HoveredParticleInfo};
use crate::core::sim::ElectrostaticSim2D;
use crate::core::boundary::BoundaryType;
use crate::core::lz2d;
use crate::presets::PresetVariant;
use crate::visual::colors::heatmap_rgb;

/// 尝试加载中文字体，返回是否成功加载
fn load_chinese_fonts(fonts: &mut egui::FontDefinitions) -> bool {
    let font_candidates = [
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\simsun.ttc",
        "C:\\Windows\\Fonts\\yahei.ttf",
        "C:\\Windows\\Fonts\\msyhbd.ttc",
    ];

    for path in &font_candidates {
        if let Ok(data) = std::fs::read(path) {
            let name = format!("chinese_{}", fonts.font_data.len());
            fonts.font_data.insert(
                name.clone(),
                Arc::new(egui::FontData::from_owned(data)),
            );
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
    sim: ElectrostaticSim2D,
    paused: bool,
    v_min: f64,
    v_max: f64,
    interaction: InteractionState,
    heatmap_texture: Option<TextureHandle>,
    show_left_panel: bool,
    show_right_panel: bool,
    show_heatmap: bool,
    show_grid: bool,
    show_about_dialog: bool,
    show_shortcuts_dialog: bool,
    message_dialog: Option<String>,
}

pub struct LiziApp {
    state: Option<SimulationState>,
}

impl Default for LiziApp {
    fn default() -> Self {
        Self { state: None }
    }
}

impl LiziApp {
    pub fn run() {
        let native_options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1100.0, 700.0]),
            ..Default::default()
        };
        let _ = eframe::run_native(
            "LiziEngine2D - 静电 PIC 模拟器",
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
                ui.heading("LiziEngine2D");
                ui.label("静电 PIC 模拟器");
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
                                    heatmap_texture: None,
                                    show_left_panel: true, show_right_panel: true,
                                    show_heatmap: true, show_grid: false,
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
                if ui.button("📂 导入场景...").clicked() {
                    ui.close_menu();
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("LiziEngine2D 场景", &["lz2d"]).pick_file()
                    {
                        match lz2d::load_from_file(path.to_string_lossy().as_ref()) {
                            Ok((loaded_sim, _)) => {
                                state.sim = loaded_sim;
                                state.paused = true;
                                state.v_min = 0.0; state.v_max = 1.0;
                                state.heatmap_texture = None;
                                state.interaction = InteractionState::new();
                                state.sim.v = None;
                                state.sim.ex = None;
                                state.sim.ey = None;
                                state.message_dialog = Some(format!("✅ 成功导入场景\n路径: {}", path.display()));
                            }
                            Err(e) => state.message_dialog = Some(format!("❌ 导入失败\n{}", e)),
                        }
                    }
                }
                if ui.button("💾 导出场景...").clicked() {
                    ui.close_menu();
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("LiziEngine2D 场景", &["lz2d"])
                        .set_file_name("scene.lz2d").save_file()
                    {
                        match lz2d::save_to_file(&state.sim, 0, path.to_string_lossy().as_ref()) {
                            Ok(()) => state.message_dialog = Some(format!("✅ 成功导出场景\n路径: {}", path.display())),
                            Err(e) => state.message_dialog = Some(format!("❌ 导出失败\n{}", e)),
                        }
                    }
                }
                ui.separator();
                if ui.button("返回预设选择").clicked() { back_requested = true; ui.close_menu(); }
                ui.separator();
                if ui.button("退出").clicked() { std::process::exit(0); }
            });
            ui.menu_button("选项", |ui| {
                let mut show_left = state.show_left_panel;
                if ui.checkbox(&mut show_left, "显示工具面板").changed() { state.show_left_panel = show_left; }
                let mut show_right = state.show_right_panel;
                if ui.checkbox(&mut show_right, "显示参数面板").changed() { state.show_right_panel = show_right; }
                ui.separator();
                let mut show_heatmap = state.show_heatmap;
                if ui.checkbox(&mut show_heatmap, "显示热力图").changed() { state.show_heatmap = show_heatmap; }
                let mut show_grid = state.show_grid;
                if ui.checkbox(&mut show_grid, "显示网格").changed() { state.show_grid = show_grid; }
            });
            ui.menu_button("帮助", |ui| {
                if ui.button("关于 LiziEngine2D").clicked() { state.show_about_dialog = true; ui.close_menu(); }
                if ui.button("快捷键说明").clicked() { state.show_shortcuts_dialog = true; ui.close_menu(); }
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!("预设: {}", state.variant.display_name()));
                if let Some(v) = state.sim.v.as_ref() {
                    let actual_min = v.iter().cloned().fold(f64::MAX, f64::min);
                    let actual_max = v.iter().cloned().fold(f64::MIN, f64::max);
                    ui.label(format!("V: [{:.2e}, {:.2e}]", actual_min, actual_max));
                }
                ui.label(format!("粒子数: {}", state.sim.particles.len()));
                ui.separator();
                if ui.button("⟳ Reset").clicked() {
                    let new_sim = state.variant.create_sim();
                    state.sim = new_sim; state.paused = false;
                    state.v_min = 0.0; state.v_max = 1.0;
                    state.interaction = InteractionState::new();
                }
                if ui.button("⏭ Step").clicked() {
                    state.paused = true;
                    let dt = state.variant.config().dt;
                    state.sim.step(dt);
                }
                if state.paused { if ui.button("▶ Play").clicked() { state.paused = false; } }
                else { if ui.button("⏸ Pause").clicked() { state.paused = true; } }
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
        egui::Window::new("关于 LiziEngine2D").open(&mut state.show_about_dialog).resizable(false)
            .default_size([420.0, 280.0]).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("LiziEngine2D"); ui.label("版本 0.1.0"); ui.separator(); ui.add_space(8.0);
                ui.label("二维静电 PIC (Particle-in-Cell) 模拟器"); ui.label("使用 Rust + egui 实现");
                ui.add_space(8.0);
                ui.hyperlink_to("GitHub 仓库", "https://github.com/TheSingularityStudio/LiziEngine2D");
                ui.add_space(8.0); ui.separator(); ui.add_space(8.0);
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
                ui.label("左侧面板选择四种工具：");
                ui.label("  • 拖动粒子 — 点击选中粒子并拖拽移动");
                ui.label("  • 放置粒子 — 点击画布空白处创建新粒子");
                ui.label("  • 删除粒子 — 点击粒子将其删除");
                ui.label("  • 查看 — 滚轮缩放、拖拽平移画布，悬停查看粒子参数");
                ui.add_space(8.0);
                ui.heading("画布操作"); ui.separator(); ui.add_space(4.0);
                ui.label("  • 鼠标左键 — 根据当前工具执行操作");
                ui.label("  • 鼠标拖拽 — 平移画布（查看模式下）");
                ui.label("  • 滚轮 — 缩放画布（查看模式下）");
                ui.label("  • 面板显示/隐藏 — 在\"选项\"菜单中控制");
                ui.add_space(8.0);
                ui.heading("模拟控制"); ui.separator(); ui.add_space(4.0);
                ui.label("  • ▶ Play — 启动自动步进模拟"); ui.label("  • ⏸ Pause — 暂停模拟");
                ui.label("  • ⏭ Step — 单步执行一个时间步"); ui.label("  • ⟳ Reset — 重置到初始状态");
                ui.label("  • ← 返回 — 返回预设选择界面");
                ui.add_space(8.0);
                ui.heading("菜单栏"); ui.separator(); ui.add_space(4.0);
                ui.label("  • 文件 → 返回预设选择 / 退出"); ui.label("  • 选项 → 显示/隐藏面板和热力图");
                ui.label("  • 帮助 → 关于 / 快捷键说明");
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
                let is_selected = interaction.tool_mode == tool;
                let text = format!("{} {}", tool.icon(), tool.display_name());
                let button = if is_selected {
                    egui::Button::new(text).fill(ui.style().visuals.selection.bg_fill).min_size(egui::vec2(120.0, 32.0))
                } else {
                    egui::Button::new(text).min_size(egui::vec2(120.0, 32.0))
                };
                if ui.add(button).clicked() { interaction.tool_mode = tool; }
            }
            ui.add_space(16.0); ui.separator(); ui.add_space(8.0);
            ui.label("快捷操作提示：");
            ui.label("拖拽可选择粒子");
            ui.label("点击画布执行操作");
        });
    });
}

fn render_right_panel(ctx: &egui::Context, state: &mut SimulationState) {
    if !state.show_right_panel { return; }
    let interaction = &mut state.interaction;
    egui::SidePanel::right("param_panel").resizable(false).default_width(180.0).show(ctx, |ui| {
        ui.vertical(|ui| {
            ui.add_space(8.0); ui.heading("参数"); ui.separator(); ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("当前工具："); ui.label(interaction.tool_mode.display_name());
            });
            ui.add_space(8.0);

            match interaction.tool_mode {
                ToolMode::DragParticle => {
                    ui.label("拖动粒子"); ui.add_space(4.0);
                    ui.label("拖拽粒子改变其位置。"); ui.label("选中时粒子速度归零。");
                    ui.add_space(8.0); ui.label("选择半径：");
                    ui.add(egui::Slider::new(&mut interaction.selection_radius, 0.01..=0.20)
                        .text("归一化").step_by(0.005));
                }
                ToolMode::PlaceParticle => {
                    let list_enabled = &mut interaction.placement_list.enabled;
                    ui.checkbox(list_enabled, "使用放置清单"); ui.add_space(4.0);
                    if *list_enabled {
                        ui.separator(); ui.add_space(4.0);
                        ui.label("放置清单（点击画布一次性放置所有粒子）："); ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label("排列方式："); ui.add_space(4.0);
                            for mode in ArrangeMode::all() {
                                ui.radio_value(&mut interaction.placement_list.arrange_mode, mode, mode.display_name());
                            }
                        });
                        ui.horizontal(|ui| { ui.set_min_width(60.0); ui.label("间距："); });
                        ui.add(egui::Slider::new(&mut interaction.placement_list.spacing, 0.005..=0.20)
                            .text("归一化").step_by(0.005));
                        ui.add_space(6.0); ui.separator(); ui.add_space(4.0);
                        let mut remove_idx: Option<usize> = None;
                        for (i, entry) in interaction.placement_list.entries.iter_mut().enumerate() {
                            ui.group(|ui| {
                                ui.horizontal(|ui| {
                                    ui.label(format!("#{}", i + 1));
                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                        if ui.button("❌").clicked() { remove_idx = Some(i); }
                                    });
                                });
                                ui.horizontal(|ui| {
                                    ui.label("电荷量：");
                                    ui.add(egui::Slider::new(&mut entry.charge, -10.0..=10.0).text("q").step_by(0.1));
                                });
                                ui.horizontal(|ui| {
                                    ui.label("质量：");
                                    ui.add(egui::Slider::new(&mut entry.mass, 0.1..=10.0).text("m").step_by(0.1));
                                });
                                ui.checkbox(&mut entry.fixed, "固定（速度=0）");
                            });
                            ui.add_space(4.0);
                        }
                        if let Some(idx) = remove_idx { interaction.placement_list.entries.remove(idx); }
                        ui.horizontal(|ui| {
                            if ui.button("+ 添加粒子").clicked() {
                                interaction.placement_list.entries.push(crate::gui::interaction::PlacementEntry::default());
                            }
                            if ui.button("- 清空清单").clicked() { interaction.placement_list.entries.clear(); }
                        });
                        // 导入/导出按钮
                        ui.horizontal(|ui| {
                            if ui.button("📥 导入清单").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("放置清单", &["json"])
                                    .pick_file()
                                {
                                    match std::fs::read_to_string(path.clone()) {
                                        Ok(content) => {
                                            match interaction.placement_list.import_json(&content) {
                                                Ok(()) => {
                                                    state.message_dialog = Some(format!("✅ 成功导入清单\n路径: {}", path.display()));
                                                }
                                                Err(e) => {
                                                    state.message_dialog = Some(format!("❌ 导入失败\n{}", e));
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            state.message_dialog = Some(format!("❌ 读取文件失败\n{}", e));
                                        }
                                    }
                                }
                            }
                            if ui.button("📤 导出清单").clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("放置清单", &["json"])
                                    .set_file_name("placement_list.json")
                                    .save_file()
                                {
                                    match interaction.placement_list.export_json() {
                                        Ok(content) => {
                                            match std::fs::write(&path, &content) {
                                                Ok(()) => {
                                                    state.message_dialog = Some(format!("✅ 成功导出清单\n路径: {}", path.display()));
                                                }
                                                Err(e) => {
                                                    state.message_dialog = Some(format!("❌ 写入文件失败\n{}", e));
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            state.message_dialog = Some(format!("❌ 导出失败\n{}", e));
                                        }
                                    }
                                }
                            }
                        });
                        ui.add_space(4.0); ui.separator(); ui.add_space(4.0);
                        ui.label(format!("清单中共 {} 个粒子", interaction.placement_list.entries.len()));
                        ui.label("点击画布放置所有粒子。");
                    } else {
                        ui.label("快速放置参数："); ui.add_space(4.0);
                        ui.horizontal(|ui| { ui.set_min_width(60.0); ui.label("电荷量："); });
                        ui.add(egui::Slider::new(&mut interaction.place_params.charge, -10.0..=10.0).text("q").step_by(0.1));
                        ui.horizontal(|ui| { ui.set_min_width(60.0); ui.label("质量："); });
                        ui.add(egui::Slider::new(&mut interaction.place_params.mass, 0.1..=10.0).text("m").step_by(0.1));
                        ui.add_space(4.0);
                        ui.checkbox(&mut interaction.place_params.fixed, "固定粒子（速度=0）");
                        ui.add_space(8.0); ui.separator(); ui.add_space(4.0);
                        ui.label("在画布上点击放置粒子。");
                        if interaction.place_params.fixed { ui.label("固定粒子不会移动。"); }
                    }
                }
                ToolMode::DeleteParticle => {
                    ui.label("删除粒子"); ui.add_space(4.0);
                    ui.label("点击粒子将其删除。"); ui.add_space(8.0); ui.label("选择半径：");
                    ui.add(egui::Slider::new(&mut interaction.selection_radius, 0.01..=0.20)
                        .text("归一化").step_by(0.005));
                }
                ToolMode::Inspect => {
                    ui.label("查看工具"); ui.add_space(4.0);
                    ui.label("滚轮缩放画布"); ui.label("鼠标拖拽平移画布");
                    ui.add_space(4.0);
                    ui.label(format!("缩放: {:.1}x", interaction.zoom));
                    ui.label(format!("偏移: ({:.1}, {:.1})", interaction.view_offset.0, interaction.view_offset.1));
                    ui.add_space(4.0);
                    if ui.button("重置视图").clicked() { interaction.reset_view(); }
                    ui.add_space(8.0); ui.separator(); ui.add_space(4.0);

                    // 显示悬停粒子信息
                    if let Some(info) = &interaction.hovered_particle {
                        ui.heading("粒子信息");
                        ui.separator();
                        ui.add_space(4.0);
                        ui.label(format!("索引: #{}", info.index));
                        ui.label(format!("位置: ({:.3}, {:.3})", info.x, info.y));
                        ui.label(format!("速度: ({:.3e}, {:.3e})", info.vx, info.vy));
                        ui.label(format!("电荷量: {:.3}", info.q));
                        ui.label(format!("质量: {:.3}", info.m));
                    } else {
                        ui.label("将鼠标悬停在粒子上");
                        ui.label("查看详细信息");
                    }
                }
            }

            ui.add_space(16.0); ui.separator(); ui.add_space(8.0);
            ui.label("模拟状态：");
            let dt = state.variant.config().dt;
            ui.label(format!("dt = {:.2e}", dt));
            if state.paused { ui.label("⏸ 已暂停"); } else { ui.label("▶ 运行中"); }

            ui.add_space(16.0); ui.separator(); ui.add_space(8.0);
            ui.label("边界类型：");
            for bt in BoundaryType::all() {
                ui.radio_value(&mut state.sim.boundary_type, bt, bt.display_name());
            }
            ui.label(match state.sim.boundary_type {
                BoundaryType::Periodic => "粒子从一边穿出，从另一边进入",
                BoundaryType::Reflective => "粒子撞到边界后反弹",
                BoundaryType::Open => "粒子移出边界即被删除",
            });

            ui.add_space(16.0); ui.separator(); ui.add_space(8.0);
            ui.label("重力设置：");
            ui.checkbox(&mut state.sim.gravity_enabled, "启用重力");
            if state.sim.gravity_enabled {
                ui.add_space(4.0);
                ui.horizontal(|ui| { ui.set_min_width(50.0); ui.label("大小："); });
                ui.add(egui::Slider::new(&mut state.sim.gravity_y, -50.0..=50.0).text("g").step_by(0.1));
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label("方向：");
                    let mut dir_idx = if state.sim.gravity_y < 0.0 { 0 } else if state.sim.gravity_y > 0.0 { 1 } else { 0 };
                    let dirs = ["↓ 向下", "↑ 向上"];
                    let changed = ui.radio_value(&mut dir_idx, 0, dirs[0]).changed()
                        || ui.radio_value(&mut dir_idx, 1, dirs[1]).changed();
                    if changed { let abs_val = state.sim.gravity_y.abs().max(1.0); state.sim.gravity_y = if dir_idx == 0 { -abs_val } else { abs_val }; }
                });
                ui.add_space(4.0); ui.label(format!("当前重力: ({:.2}, {:.2})", state.sim.gravity_x, state.sim.gravity_y));
            }

            ui.add_space(16.0); ui.separator(); ui.add_space(8.0);
            ui.label("摩擦力设置："); ui.checkbox(&mut state.sim.friction_enabled, "启用摩擦力");
            if state.sim.friction_enabled {
                ui.add_space(4.0);
                ui.horizontal(|ui| { ui.set_min_width(50.0); ui.label("阻尼："); });
                ui.add(egui::Slider::new(&mut state.sim.friction_damping, 0.0..=5.0).text("系数").step_by(0.01));
                ui.add_space(4.0); ui.label("F = -damping × v"); ui.label("值越大，粒子减速越快");
            }
        });
    });
}

fn render_simulation_panels(ctx: &egui::Context, state: &mut SimulationState) -> bool {
    let back_flag = render_menu_bar(ctx, state);
    render_dialogs(ctx, state);
    render_left_panel(ctx, state);
    render_right_panel(ctx, state);
    render_central_canvas(ctx, state);
    if !state.paused {
        let dt = state.variant.config().dt;
        state.sim.step(dt);
        ctx.request_repaint();
    }
    !back_flag
}

fn render_central_canvas(ctx: &egui::Context, state: &mut SimulationState) {
    let sim = &mut state.sim;
    let v_min = &mut state.v_min;
    let v_max = &mut state.v_max;
    let interaction = &mut state.interaction;

    egui::CentralPanel::default().show(ctx, |ui| {
        if sim.v.is_none() || sim.ex.is_none() || sim.ey.is_none() {
            sim.compute_fields();
        }

        let snapshot = sim.get_state_snapshot();
        let (nx, ny) = snapshot.v.dim();

        let mut min = f64::MAX;
        let mut max = f64::MIN;
        for val in snapshot.v.iter() {
            if *val < min { min = *val; }
            if *val > max { max = *val; }
        }
        *v_min = *v_min * 0.9 + min * 0.1;
        *v_max = *v_max * 0.9 + max * 0.1;
        if (*v_max - *v_min).abs() < 1e-12 { *v_max = *v_min + 1.0; }

        let avail = ui.available_rect_before_wrap();
        let max_edge = avail.size().x.min(avail.size().y);
        let center = avail.center();
        let image_rect = egui::Rect::from_center_size(center, egui::vec2(max_edge, max_edge));

        // 应用查看工具的平移和缩放
        let texture_rect: egui::Rect;
        let zoom = interaction.zoom.max(0.1).min(10.0);
        if interaction.tool_mode == ToolMode::Inspect {
            let off_x = interaction.view_offset.0;
            let off_y = interaction.view_offset.1;
            let zoomed_size = max_edge * zoom;
            let zoomed_rect = egui::Rect::from_center_size(
                egui::pos2(center.x + off_x, center.y + off_y),
                egui::vec2(zoomed_size, zoomed_size),
            );
            texture_rect = zoomed_rect;
        } else {
            texture_rect = image_rect;
        }

        if state.show_heatmap {
            let mut pixels = Vec::with_capacity(nx * ny);
            for j in (0..ny).rev() {
                for i in 0..nx {
                    let val = snapshot.v[[i, j]];
                    let (r, g, b) = heatmap_rgb(val, *v_min, *v_max);
                    pixels.push(egui::Color32::from_rgb(r, g, b));
                }
            }
            let color_image = ColorImage { size: [nx, ny], pixels };
            let texture = state.heatmap_texture.get_or_insert_with(|| {
                ctx.load_texture("heatmap", color_image.clone(), egui::TextureOptions::NEAREST)
            });
            texture.set(color_image, egui::TextureOptions::NEAREST);
            let _response = ui.put(
                texture_rect,
                egui::Image::from_texture(SizedTexture::from(&*texture))
                    .fit_to_exact_size(texture_rect.size()),
            );
        } else {
            let bg_color = ui.style().visuals.panel_fill;
            let painter = ui.painter();
            painter.rect_filled(texture_rect, 0.0, bg_color);
            painter.rect_stroke(texture_rect, 0.0, (1.0, egui::Color32::GRAY), egui::StrokeKind::Inside);
        }

        // 绘制粒子
        let painter = ui.painter();

        // 绘制网格（如果启用）
        if state.show_grid {
            let grid_color = egui::Color32::from_gray(100);
            let cell_w = texture_rect.width() / nx as f32;
            let cell_h = texture_rect.height() / ny as f32;
            // 垂直线
            for i in 0..=nx {
                let x = texture_rect.left() + i as f32 * cell_w;
                painter.line_segment(
                    [egui::pos2(x, texture_rect.top()), egui::pos2(x, texture_rect.bottom())],
                    (0.5, grid_color),
                );
            }
            // 水平线
            for j in 0..=ny {
                let y = texture_rect.bottom() - j as f32 * cell_h;
                painter.line_segment(
                    [egui::pos2(texture_rect.left(), y), egui::pos2(texture_rect.right(), y)],
                    (0.5, grid_color),
                );
            }
        }
        let particle_count = snapshot.x.len();
        let lx = if snapshot.lx <= 0.0 { 1.0 } else { snapshot.lx };
        let ly = if snapshot.ly <= 0.0 { 1.0 } else { snapshot.ly };

        for p in 0..particle_count {
            let nx_p = (((snapshot.x[p] / lx) * nx as f64 + 0.5) / nx as f64).clamp(0.0, 1.0);
            let ny_p = (((snapshot.y[p] / ly) * ny as f64 + 0.5) / ny as f64).clamp(0.0, 1.0);

            let sx = texture_rect.left() + nx_p as f32 * texture_rect.width();
            let sy = texture_rect.bottom() - ny_p as f32 * texture_rect.height();

            let color = if snapshot.q[p] < 0.0 { egui::Color32::CYAN } else { egui::Color32::WHITE };

            // 高亮悬停的粒子
            let radius = if interaction.tool_mode == ToolMode::Inspect {
                if let Some(info) = &interaction.hovered_particle {
                    if info.index == p { 5.0 } else { 3.0 }
                } else { 3.0 }
            } else { 3.0 };

            painter.circle_filled(egui::pos2(sx, sy), radius, color);

            // 绘制高亮圈
            if interaction.tool_mode == ToolMode::Inspect {
                if let Some(info) = &interaction.hovered_particle {
                    if info.index == p {
                        painter.circle_stroke(egui::pos2(sx, sy), 7.0, (2.0, egui::Color32::YELLOW));
                    }
                }
            }
        }

        // 查看工具：粒子悬停信息（使用屏幕坐标画信息框）
        if interaction.tool_mode == ToolMode::Inspect {
            if let Some(info) = &interaction.hovered_particle.clone() {
                if info.index < particle_count {
                    let nx_p = (((snapshot.x[info.index] / lx) * nx as f64 + 0.5) / nx as f64).clamp(0.0, 1.0);
                    let ny_p = (((snapshot.y[info.index] / ly) * ny as f64 + 0.5) / ny as f64).clamp(0.0, 1.0);
                    let sx = texture_rect.left() + nx_p as f32 * texture_rect.width();
                    let sy = texture_rect.bottom() - ny_p as f32 * texture_rect.height();

                    let tooltip_pos = egui::pos2(sx + 10.0, sy - 40.0);
                    let text = format!(
                        "#{}  q={:.2}  m={:.2}\n({:.3}, {:.3})  v=({:.2e}, {:.2e})",
                        info.index, info.q, info.m, info.x, info.y, info.vx, info.vy
                    );
                    painter.text(
                        tooltip_pos,
                        egui::Align2::LEFT_TOP,
                        text,
                        egui::TextStyle::Body.resolve(ui.style()),
                        egui::Color32::YELLOW,
                    );
                }
            }
        }

        // 处理鼠标交互
        handle_mouse_interaction(ui, sim, interaction, texture_rect, nx, ny, lx, ly);
    });
}

fn handle_mouse_interaction(
    ui: &egui::Ui,
    sim: &mut ElectrostaticSim2D,
    interaction: &mut InteractionState,
    texture_rect: egui::Rect,
    grid_nx: usize,
    grid_ny: usize,
    lx: f64,
    ly: f64,
) {
    let mouse_pos = ui.input(|i| i.pointer.hover_pos());
    let mouse_down = ui.input(|i| i.pointer.any_down());
    let mouse_clicked = ui.input(|i| i.pointer.any_click());
    let scroll_delta = ui.input(|i| i.raw_scroll_delta);

    let Some(pos) = mouse_pos else {
        if interaction.tool_mode == ToolMode::Inspect {
            interaction.hovered_particle = None;
        }
        interaction.dragging = false;
        interaction.dragged_particle_index = None;
        return;
    };

    if !texture_rect.contains(pos) {
        if interaction.tool_mode == ToolMode::Inspect {
            interaction.hovered_particle = None;
        }
        interaction.dragging = false;
        interaction.dragged_particle_index = None;
        return;
    }

    let tex_u = ((pos.x - texture_rect.left()) / texture_rect.width()).clamp(0.0f32, 1.0f32);
    let tex_v = ((texture_rect.bottom() - pos.y) / texture_rect.height()).clamp(0.0f32, 1.0f32);
    let inv_nx = grid_nx as f64;
    let inv_ny = grid_ny as f64;
    let tex_u_f64 = tex_u as f64;
    let tex_v_f64 = tex_v as f64;
    let world_x = ((tex_u_f64 * inv_nx - 0.5) / inv_nx).clamp(0.0, 1.0) * lx;
    let world_y = ((tex_v_f64 * inv_ny - 0.5) / inv_ny).clamp(0.0, 1.0) * ly;

    match interaction.tool_mode {
        ToolMode::DragParticle => {
            handle_drag_interaction(
                ui, sim, interaction, texture_rect, grid_nx, grid_ny, lx, ly,
                pos, tex_u, tex_v, tex_u_f64, tex_v_f64, world_x, world_y, mouse_down,
            );
            if interaction.dragging && mouse_down { sim.v = None; sim.ex = None; sim.ey = None; }
        }
        ToolMode::PlaceParticle => {
            if mouse_clicked {
                if interaction.placement_list.enabled && !interaction.placement_list.entries.is_empty() {
                    let offsets = interaction.compute_placement_offsets();
                    for (i, entry) in interaction.placement_list.entries.iter().enumerate() {
                        let (dx, dy) = offsets.get(i).copied().unwrap_or((0.0, 0.0));
                        let px = (world_x + dx).clamp(0.0, lx);
                        let py = (world_y + dy).clamp(0.0, ly);
                        let vx = if entry.fixed { 0.0 } else { 0.0 };
                        let vy = if entry.fixed { 0.0 } else { 0.0 };
                        sim.particles.add_particle(px, py, entry.charge, entry.mass, vx, vy);
                    }
                    sim.v = None; sim.ex = None; sim.ey = None;
                    ui.ctx().request_repaint();
                } else {
                    let charge = interaction.place_params.charge;
                    let mass = interaction.place_params.mass;
                    let fixed = interaction.place_params.fixed;
                    let vx = if fixed { 0.0 } else { 0.0 };
                    let vy = if fixed { 0.0 } else { 0.0 };
                    sim.particles.add_particle(world_x, world_y, charge, mass, vx, vy);
                    sim.v = None; sim.ex = None; sim.ey = None;
                    ui.ctx().request_repaint();
                }
            }
        }
        ToolMode::DeleteParticle => {
            if mouse_clicked {
                let particle_visual_u: Vec<f64> = sim.particles.x.iter()
                    .map(|&x| (((x / lx) * inv_nx + 0.5) / inv_nx).clamp(0.0, 1.0)).collect();
                let particle_visual_v: Vec<f64> = sim.particles.y.iter()
                    .map(|&y| (((y / ly) * inv_ny + 0.5) / inv_ny).clamp(0.0, 1.0)).collect();
                let mut min_dist = f64::MAX;
                let mut min_index = None;
                for i in 0..sim.particles.len() {
                    let dx = particle_visual_u[i] - tex_u as f64;
                    let dy = particle_visual_v[i] - tex_v as f64;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist < min_dist { min_dist = dist; min_index = Some(i); }
                }
                if let Some(idx) = min_index {
                    if min_dist <= interaction.selection_radius {
                        sim.particles.remove_particle(idx);
                        sim.v = None; sim.ex = None; sim.ey = None;
                        ui.ctx().request_repaint();
                    }
                }
            }
        }
        ToolMode::Inspect => {
            // 滚轮缩放
            if scroll_delta.y != 0.0 {
                let zoom_factor = if scroll_delta.y > 0.0 { 1.1 } else { 1.0 / 1.1 };
                let old_zoom = interaction.zoom;
                interaction.zoom = (interaction.zoom * zoom_factor).clamp(0.1, 10.0);
                // 以鼠标位置为中心缩放：调整偏移量
                let actual_zoom_change = interaction.zoom / old_zoom;
                let mouse_to_center_x = pos.x - (texture_rect.center().x - interaction.view_offset.0);
                let mouse_to_center_y = pos.y - (texture_rect.center().y - interaction.view_offset.1);
                interaction.view_offset.0 -= mouse_to_center_x * (actual_zoom_change - 1.0);
                interaction.view_offset.1 -= mouse_to_center_y * (actual_zoom_change - 1.0);
                ui.ctx().request_repaint();
            }

            // 鼠标拖拽平移（左键/右键/中键均可）
            let any_pan_button = mouse_down && (
                ui.input(|i| i.pointer.button_down(egui::PointerButton::Primary))
                || ui.input(|i| i.pointer.button_down(egui::PointerButton::Secondary))
                || ui.input(|i| i.pointer.button_down(egui::PointerButton::Middle))
            );
            if any_pan_button {
                if !interaction.panning {
                    interaction.panning = true;
                    interaction.last_pan_pos = Some((pos.x, pos.y));
                } else if let Some(last_pos) = interaction.last_pan_pos {
                    let dx = pos.x - last_pos.0;
                    let dy = pos.y - last_pos.1;
                    interaction.view_offset.0 += dx;
                    interaction.view_offset.1 += dy;
                    interaction.last_pan_pos = Some((pos.x, pos.y));
                    ui.ctx().request_repaint();
                }
            } else {
                interaction.panning = false;
                interaction.last_pan_pos = None;
            }

            // 悬停粒子检测
            if !mouse_down {
                let particle_visual_u: Vec<f64> = sim.particles.x.iter()
                    .map(|&x| (((x / lx) * inv_nx + 0.5) / inv_nx).clamp(0.0, 1.0)).collect();
                let particle_visual_v: Vec<f64> = sim.particles.y.iter()
                    .map(|&y| (((y / ly) * inv_ny + 0.5) / inv_ny).clamp(0.0, 1.0)).collect();

                let mut min_dist = f64::MAX;
                let mut min_index = None;
                for i in 0..sim.particles.len() {
                    let dx = particle_visual_u[i] - tex_u as f64;
                    let dy = particle_visual_v[i] - tex_v as f64;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist < min_dist { min_dist = dist; min_index = Some(i); }
                }

                if let Some(idx) = min_index {
                    if min_dist <= interaction.selection_radius {
                        interaction.hovered_particle = Some(HoveredParticleInfo {
                            index: idx,
                            x: sim.particles.x[idx],
                            y: sim.particles.y[idx],
                            vx: sim.particles.vx[idx],
                            vy: sim.particles.vy[idx],
                            q: sim.particles.q[idx],
                            m: sim.particles.m[idx],
                        });
                        ui.ctx().request_repaint();
                    } else {
                        interaction.hovered_particle = None;
                    }
                } else {
                    interaction.hovered_particle = None;
                }
            }

            // 不处理鼠标点击其他操作
        }
    }

    if !mouse_down {
        interaction.dragging = false;
        interaction.dragged_particle_index = None;
    }
}

fn handle_drag_interaction(
    _ui: &egui::Ui,
    sim: &mut ElectrostaticSim2D,
    interaction: &mut InteractionState,
    _texture_rect: egui::Rect,
    grid_nx: usize,
    grid_ny: usize,
    lx: f64,
    ly: f64,
    _pos: egui::Pos2,
    tex_u: f32,
    tex_v: f32,
    _tex_u_f64: f64,
    _tex_v_f64: f64,
    world_x: f64,
    world_y: f64,
    mouse_down: bool,
) {
    let inv_nx = grid_nx as f64;
    let inv_ny = grid_ny as f64;

    if mouse_down {
        if !interaction.dragging {
            let particle_visual_u: Vec<f64> = sim.particles.x.iter()
                .map(|&x| (((x / lx) * inv_nx + 0.5) / inv_nx).clamp(0.0, 1.0)).collect();
            let particle_visual_v: Vec<f64> = sim.particles.y.iter()
                .map(|&y| (((y / ly) * inv_ny + 0.5) / inv_ny).clamp(0.0, 1.0)).collect();
            let mut min_dist = f64::MAX;
            let mut min_index = None;
            for i in 0..sim.particles.len() {
                let dx = particle_visual_u[i] - tex_u as f64;
                let dy = particle_visual_v[i] - tex_v as f64;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < min_dist { min_dist = dist; min_index = Some(i); }
            }
            if let Some(idx) = min_index {
                if min_dist <= interaction.selection_radius {
                    interaction.dragging = true;
                    interaction.dragged_particle_index = Some(idx);
                }
            }
        }
        if let Some(idx) = interaction.dragged_particle_index {
            sim.particles.x[idx] = world_x;
            sim.particles.y[idx] = world_y;
            sim.particles.vx[idx] = 0.0;
            sim.particles.vy[idx] = 0.0;
        }
    } else {
        interaction.dragging = false;
        interaction.dragged_particle_index = None;
    }
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