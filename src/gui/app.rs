use std::sync::Arc;
use eframe::egui;
use egui::menu;

use crate::gui::gl_renderer::GlRenderer;
use crate::gui::interaction::{InteractionState, ToolMode, HoveredParticleInfo, OrbitCamera};
use crate::core::sim::ElectrostaticSim3D;
use crate::core::boundary::BoundaryType;
use crate::presets::PresetVariant;

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
    show_heatmap: bool,
    show_grid: bool,
    show_axes: bool,
    show_about_dialog: bool,
    show_shortcuts_dialog: bool,
    message_dialog: Option<String>,
    gl_renderer: Option<Arc<GlRenderer>>,
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
        let _ = eframe::run_native("LiziEngine3D - 静电 PIC 模拟器", native_options,
            Box::new(|cc| {
                let mut fonts = egui::FontDefinitions::default();
                if !load_chinese_fonts(&mut fonts) {
                    eprintln!("warning: no chinese fonts");
                }
                cc.egui_ctx.set_fonts(fonts);

                // Initialize GlRenderer with glow context
                Ok(Box::new(LiziApp {
                    state: None,
                }))
            }),
        );
    }

    fn render_preset_selection(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(60.0);
                ui.heading("LiziEngine3D");
                ui.label("3D PIC Simulator");
                ui.add_space(10.0); ui.separator(); ui.add_space(20.0);
                ui.label("Select a preset:");
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
                                    variant: *variant, sim, paused: false,
                                    v_min: 0.0, v_max: 1.0,
                                    interaction: InteractionState::new(),
                                    show_left_panel: true, show_right_panel: true,
                                    show_heatmap: true, show_grid: false, show_axes: true,
                                    show_about_dialog: false, show_shortcuts_dialog: false,
                                    message_dialog: None,
                                    gl_renderer: None,
                                });
                            }
                            ui.add_space(10.0); ui.label(desc);
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
            ui.menu_button("File", |ui| {
                if ui.button("Back to Presets").clicked() { back_requested = true; ui.close_menu(); }
                ui.separator();
                if ui.button("Exit").clicked() { std::process::exit(0); }
            });
            ui.menu_button("Options", |ui| {
                let mut sl = state.show_left_panel;
                if ui.checkbox(&mut sl, "Show Tools").changed() { state.show_left_panel = sl; }
                let mut sr = state.show_right_panel;
                if ui.checkbox(&mut sr, "Show Params").changed() { state.show_right_panel = sr; }
                ui.separator();
                let mut sh = state.show_heatmap;
                if ui.checkbox(&mut sh, "Show Heatmap").changed() { state.show_heatmap = sh; }
                let mut sg = state.show_grid;
                if ui.checkbox(&mut sg, "Show Grid").changed() { state.show_grid = sg; }
                let mut sa = state.show_axes;
                if ui.checkbox(&mut sa, "Show Axes").changed() { state.show_axes = sa; }
            });
            ui.menu_button("Help", |ui| {
                if ui.button("About").clicked() { state.show_about_dialog = true; ui.close_menu(); }
                if ui.button("Shortcuts").clicked() { state.show_shortcuts_dialog = true; ui.close_menu(); }
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(format!("{}", state.variant.display_name()));
                ui.label(format!("N={}", state.sim.particles.len()));
                ui.separator();
                if ui.button("Reset").clicked() {
                    let ns = state.variant.create_sim();
                    state.sim = ns; state.paused = false;
                    state.v_min = 0.0; state.v_max = 1.0;
                    state.interaction = InteractionState::new();
                }
                if ui.button("Step").clicked() { state.paused = true; state.sim.step(state.variant.config().dt); }
                if state.paused { if ui.button("Play").clicked() { state.paused = false; } }
                else { if ui.button("Pause").clicked() { state.paused = true; } }
                if ui.button("Back").clicked() { back_requested = true; }
            });
        });
    });
    back_requested
}

fn render_dialogs(ctx: &egui::Context, state: &mut SimulationState) {
    if let Some(msg) = &state.message_dialog.clone() {
        let mut open = true;
        egui::Window::new("Message").open(&mut open).resizable(false).default_size([400.0, 150.0]).show(ctx, |ui| {
            ui.add_space(8.0); ui.label(msg); ui.add_space(12.0);
            if ui.button("OK").clicked() { state.message_dialog = None; }
        });
        if !open { state.message_dialog = None; }
    }
    if state.show_about_dialog {
        egui::Window::new("About LiziEngine3D").open(&mut state.show_about_dialog).resizable(false)
            .default_size([420.0, 280.0]).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("LiziEngine3D v0.1.0");
                ui.label("3D Electrostatic PIC Simulator");
                ui.label("Rust + egui + ndrustfft");
                ui.add_space(8.0);
                ui.hyperlink_to("GitHub", "https://github.com/TheSingularityStudio/LiziEngine3D");
            });
        });
    }
    if state.show_shortcuts_dialog {
        egui::Window::new("Shortcuts").open(&mut state.show_shortcuts_dialog).resizable(false)
            .default_size([380.0, 300.0]).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Tools"); ui.separator();
                ui.label("  Drag - click & drag particles");
                ui.label("  Place - click to create particle");
                ui.label("  Delete - click to remove particle");
                ui.label("  Inspect - rotate/zoom/pan 3D view");
                ui.add_space(8.0);
                ui.heading("3D Controls"); ui.separator();
                ui.label("  Left drag - rotate");
                ui.label("  Scroll - zoom");
                ui.label("  Right drag - pan");
            });
        });
    }
}

fn render_left_panel(ctx: &egui::Context, state: &mut SimulationState) {
    if !state.show_left_panel { return; }
    let interaction = &mut state.interaction;
    egui::SidePanel::left("tool_panel").resizable(false).default_width(140.0).show(ctx, |ui| {
        ui.vertical(|ui| {
            ui.add_space(8.0); ui.heading("Tools"); ui.separator(); ui.add_space(4.0);
            for tool in ToolMode::all() {
                let selected = interaction.tool_mode == tool;
                let text = format!("{} {}", tool.icon(), tool.display_name());
                let btn = if selected {
                    egui::Button::new(text).fill(ui.style().visuals.selection.bg_fill).min_size(egui::vec2(120.0, 32.0))
                } else { egui::Button::new(text).min_size(egui::vec2(120.0, 32.0)) };
                if ui.add(btn).clicked() { interaction.tool_mode = tool; }
            }
        });
    });
}

fn render_right_panel(ctx: &egui::Context, state: &mut SimulationState) {
    if !state.show_right_panel { return; }
    let interaction = &mut state.interaction;
    egui::SidePanel::right("param_panel").resizable(false).default_width(200.0).show(ctx, |ui| {
        ui.vertical(|ui| {
            ui.add_space(8.0); ui.heading("Params"); ui.separator();
            ui.label(format!("Tool: {}", interaction.tool_mode.display_name()));
            match interaction.tool_mode {
                ToolMode::DragParticle => {
                    ui.add(egui::Slider::new(&mut interaction.selection_radius, 0.01..=0.20).text("radius"));
                }
                ToolMode::PlaceParticle => {
                    ui.add(egui::Slider::new(&mut interaction.place_params.charge, -10.0..=10.0).text("charge q"));
                    ui.add(egui::Slider::new(&mut interaction.place_params.mass, 0.1..=10.0).text("mass m"));
                    ui.checkbox(&mut interaction.place_params.fixed, "fixed");
                }
                ToolMode::DeleteParticle => {
                    ui.add(egui::Slider::new(&mut interaction.selection_radius, 0.01..=0.20).text("radius"));
                }
                ToolMode::Inspect => {
                    ui.label(format!("az: {:.1}DEG", interaction.camera.azimuth.to_degrees()));
                    ui.label(format!("el: {:.1}DEG", interaction.camera.elevation.to_degrees()));
                    ui.label(format!("dist: {:.2}", interaction.camera.distance));
                    if ui.button("Reset View").clicked() { interaction.reset_view(); }
                    ui.add_space(8.0); ui.separator();
                    if let Some(info) = &interaction.hovered_particle {
                        ui.label(format!("#{} pos=({:.2},{:.2},{:.2})", info.index, info.x, info.y, info.z));
                        ui.label(format!("q={:.2} m={:.2} v=({:.2e},{:.2e},{:.2e})", info.q, info.m, info.vx, info.vy, info.vz));
                    }
                }
            }
            ui.add_space(8.0); ui.separator();
            ui.label(format!("dt={:.2e}", state.variant.config().dt));
            if state.paused { ui.label("PAUSED"); } else { ui.label("RUNNING"); }
            ui.add_space(8.0); ui.separator();
            for bt in BoundaryType::all() {
                ui.radio_value(&mut state.sim.boundary_type, bt, bt.display_name());
            }
            ui.add_space(8.0); ui.separator();
            ui.checkbox(&mut state.sim.gravity_enabled, "Gravity");
            if state.sim.gravity_enabled {
                ui.add(egui::Slider::new(&mut state.sim.gravity_z, -50.0..=50.0).text("g_z"));
            }
            ui.checkbox(&mut state.sim.friction_enabled, "Friction");
            if state.sim.friction_enabled {
                ui.add(egui::Slider::new(&mut state.sim.friction_damping, 0.0..=5.0).text("damping"));
            }
        });
    });
}

/// 3D projection for interaction (hover/drag/place/delete).
/// Screen coordinates: x right, y down. Returns (sx, sy, depth).
fn project_3d(
    wx: f64, wy: f64, wz: f64,
    cam: &OrbitCamera,
    lx: f64, ly: f64, lz: f64,
) -> Option<(f32, f32, f32)> {
    let nx = (wx / lx) as f32;
    let ny = (wy / ly) as f32;
    let nz = (wz / lz) as f32;
    let mut px = nx - cam.target_x;
    let mut py = ny - cam.target_y;
    let mut pz = nz - cam.target_z;
    let d = cam.distance.max(0.1);
    let (ca, sa) = cam.azimuth.sin_cos();
    let (ce, se) = cam.elevation.sin_cos();
    // rotate around Y (azimuth)
    let rx = px * ca - pz * sa;
    let rz = px * sa + pz * ca;
    px = rx; pz = rz;
    // rotate around X (elevation)
    let ry = py * ce - pz * se;
    let rz2 = py * se + pz * ce;
    py = ry; pz = rz2;
    let zc = (-pz + d).max(0.01);
    let scale = 2.0 / zc;
    Some((px * scale, -py * scale, zc))
}

fn render_simulation_canvas(
    ctx: &egui::Context,
    gl: Option<&std::sync::Arc<eframe::glow::Context>>,
    state: &mut SimulationState,
) {
    let sim = &mut state.sim;
    let v_min = &mut state.v_min;
    let v_max = &mut state.v_max;
    let interaction = &mut state.interaction;

    egui::CentralPanel::default().show(ctx, |ui| {
        let (rect, _response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click());
        let painter = ui.painter();
        let bg = ui.style().visuals.panel_fill;

        let cx = rect.center();
        let size = rect.width().min(rect.height()) * 0.95;
        let half = size * 0.5;

        if sim.v.is_none() || sim.ex.is_none() || sim.ey.is_none() || sim.ez.is_none() {
            sim.compute_fields();
        }
        let snapshot = sim.get_state_snapshot();
        let (_nx, _ny, _nz) = snapshot.v.dim();

        let mut min_v = f64::MAX; let mut max_v = f64::MIN;
        for val in snapshot.v.iter() { if *val < min_v { min_v = *val; } if *val > max_v { max_v = *val; } }
        *v_min = *v_min * 0.9 + min_v * 0.1;
        *v_max = *v_max * 0.9 + max_v * 0.1;
        if (*v_max - *v_min).abs() < 1e-12 { *v_max = *v_min + 1.0; }

        // ---- GPU 3D rendering via PaintCallback ----
        if let (Some(gl_ctx), Some(ref renderer)) = (gl, &state.gl_renderer) {
            // Update heatmap texture on GPU
            let v_min_copy = *v_min;
            let v_max_copy = *v_max;
            let snap_v = snapshot.v.clone();
            if state.show_heatmap {
                renderer.update_heatmap(gl_ctx, &snap_v, v_min_copy, v_max_copy);
            }

            // Clone snapshot for GPU callback (original stays for CPU interaction)
            let snapshot_for_gpu = snapshot.clone();
            let camera = interaction.camera.clone();
            let show_heatmap = state.show_heatmap;
            let show_grid = state.show_grid;
            let show_axes = state.show_axes;
            let renderer = state.gl_renderer.clone().unwrap();

            let cb = egui_glow::CallbackFn::new(move |_info, egui_painter| {
                let gl = egui_painter.gl();
                renderer.render_scene(
                    gl,
                    &snapshot_for_gpu,
                    &camera,
                    show_heatmap,
                    show_grid,
                    show_axes,
                    (rect.min.x, rect.min.y, rect.width(), rect.height()),
                );
            });

            painter.add(egui::Shape::Callback(egui::PaintCallback {
                rect,
                callback: std::sync::Arc::new(cb),
            }));
        } else {
            // Fallback: draw background rectangle only
            painter.rect_filled(rect, 0.0, bg);
        }

        // ---- Mouse & Interaction (CPU-side) ----
        let mouse_pos = ctx.input(|i| i.pointer.hover_pos());
        let mouse_down = ctx.input(|i| i.pointer.any_down());
        let mp = ctx.input(|i| i.pointer.button_down(egui::PointerButton::Primary));
        let ms = ctx.input(|i| i.pointer.button_down(egui::PointerButton::Secondary));
        let scroll = ctx.input(|i| i.raw_scroll_delta);

        let cam = &mut interaction.camera;
        if interaction.tool_mode == ToolMode::Inspect {
            if let Some(pos) = mouse_pos {
                if rect.contains(pos) {
                    if mp && !ms {
                        if let Some(last) = cam.last_mouse_pos {
                            cam.azimuth += (pos.x - last.0) * 0.008;
                            cam.elevation = (cam.elevation + (pos.y - last.1) * 0.008).clamp(-1.5, 1.5);
                        }
                        cam.last_mouse_pos = Some((pos.x, pos.y));
                    } else if ms {
                        if let Some(last) = cam.last_mouse_pos {
                            let dx = (pos.x - last.0) * 0.005 * cam.distance;
                            let dy = (pos.y - last.1) * 0.005 * cam.distance;
                            let (ca, sa) = cam.azimuth.sin_cos();
                            cam.target_x -= dx * ca; cam.target_z -= dx * sa; cam.target_y += dy;
                        }
                        cam.last_mouse_pos = Some((pos.x, pos.y));
                    } else { cam.last_mouse_pos = None; }
                    if scroll.y != 0.0 {
                        cam.distance = (cam.distance * if scroll.y > 0.0 { 0.9 } else { 1.1 }).clamp(0.3, 15.0);
                    }
                }
            }
        }

        // Pre-compute particle projections for interaction (CPU)
        struct PDraw { sx: f32, sy: f32, depth: f32, idx: usize }
        let mut draws: Vec<PDraw> = Vec::new();
        for i in 0..snapshot.x.len() {
            if let Some((sx, sy, depth)) = project_3d(snapshot.x[i], snapshot.y[i], snapshot.z[i], cam, snapshot.lx, snapshot.ly, snapshot.lz) {
                draws.push(PDraw { sx: cx.x + sx * half, sy: cx.y + sy * half, depth, idx: i });
            }
        }
        draws.sort_by(|a, b| b.depth.partial_cmp(&a.depth).unwrap_or(std::cmp::Ordering::Equal));

        // Hover detection (Inspect mode)
        if interaction.tool_mode == ToolMode::Inspect && !mouse_down {
            interaction.hovered_particle = None;
            if let Some(pos) = mouse_pos {
                if rect.contains(pos) {
                    let mut best = (interaction.selection_radius * size as f64).powi(2) as f32;
                    for d in &draws {
                        let d2 = (d.sx - pos.x).powi(2) + (d.sy - pos.y).powi(2);
                        if d2 < best { best = d2;
                            interaction.hovered_particle = Some(HoveredParticleInfo {
                                index: d.idx, x: snapshot.x[d.idx], y: snapshot.y[d.idx], z: snapshot.z[d.idx],
                                vx: snapshot.vx[d.idx], vy: snapshot.vy[d.idx], vz: snapshot.vz[d.idx],
                                q: snapshot.q[d.idx], m: snapshot.m[d.idx],
                            });
                        }
                    }
                }
            }
        }

        // CPU overlays: hover highlight + tooltip
        if let Some(h) = &interaction.hovered_particle {
            for d in &draws {
                if d.idx == h.index {
                    painter.circle_stroke(egui::pos2(d.sx, d.sy), 8.0, (2.0, egui::Color32::YELLOW));
                    let tip = egui::pos2(d.sx + 10.0, d.sy - 30.0);
                    painter.text(tip, egui::Align2::LEFT_TOP,
                        format!("#{} q={:.2} ({:.3},{:.3},{:.3})", h.index, h.q, h.x, h.y, h.z),
                        egui::TextStyle::Body.resolve(ui.style()), egui::Color32::YELLOW);
                    break;
                }
            }
        }

        // Drag interaction
        if interaction.tool_mode == ToolMode::DragParticle && mouse_down && mp {
            if let Some(pos) = mouse_pos {
                if rect.contains(pos) {
                    if !interaction.dragging {
                        let mut best = interaction.selection_radius * size as f64;
                        for d in &draws {
                            let dist = (((d.sx - pos.x) as f64).powi(2) + ((d.sy - pos.y) as f64).powi(2)).sqrt();
                            if dist < best { best = dist; interaction.dragged_particle_index = Some(d.idx); interaction.dragging = true; }
                        }
                    }
                    if let Some(idx) = interaction.dragged_particle_index {
                        let nx = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0) as f64;
                        let ny = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0) as f64;
                        sim.particles.x[idx] = (nx * snapshot.lx).clamp(0.0, snapshot.lx);
                        sim.particles.y[idx] = ((1.0 - ny) * snapshot.ly).clamp(0.0, snapshot.ly);
                        sim.particles.vx[idx] = 0.0; sim.particles.vy[idx] = 0.0;
                        sim.v = None; sim.ex = None; sim.ey = None; sim.ez = None;
                    }
                }
            }
        }

        // Place particle
        if interaction.tool_mode == ToolMode::PlaceParticle && ctx.input(|i| i.pointer.any_click()) {
            if let Some(pos) = mouse_pos {
                if rect.contains(pos) {
                    let nx = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0) as f64;
                    let ny = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0) as f64;
                    sim.particles.add_particle(
                        (nx * snapshot.lx).clamp(0.0, snapshot.lx),
                        ((1.0 - ny) * snapshot.ly).clamp(0.0, snapshot.ly),
                        0.5 * snapshot.lz,
                        interaction.place_params.charge, interaction.place_params.mass, 0.0, 0.0, 0.0,
                    );
                    sim.v = None; sim.ex = None; sim.ey = None; sim.ez = None;
                }
            }
        }

        // Delete particle
        if interaction.tool_mode == ToolMode::DeleteParticle && ctx.input(|i| i.pointer.any_click()) {
            if let Some(pos) = mouse_pos {
                if rect.contains(pos) {
                    let mut best = interaction.selection_radius * size as f64;
                    let mut bi = None;
                    for d in &draws {
                        let dist = (((d.sx - pos.x) as f64).powi(2) + ((d.sy - pos.y) as f64).powi(2)).sqrt();
                        if dist < best { best = dist; bi = Some(d.idx); }
                    }
                    if let Some(idx) = bi { sim.particles.remove_particle(idx); sim.v = None; sim.ex = None; sim.ey = None; sim.ez = None; }
                }
            }
        }

        if !mouse_down { interaction.dragging = false; interaction.dragged_particle_index = None; }

        // Border (drawn last, on top)
        painter.rect_stroke(rect, 0.0, (1.0, egui::Color32::GRAY), egui::StrokeKind::Inside);
    });
}

fn render_simulation_panels(
    ctx: &egui::Context,
    gl: Option<&std::sync::Arc<eframe::glow::Context>>,
    state: &mut SimulationState,
) -> bool {
    let back = render_menu_bar(ctx, state);
    render_dialogs(ctx, state);
    render_left_panel(ctx, state);
    render_right_panel(ctx, state);
    render_simulation_canvas(ctx, gl, state);
    if !state.paused { state.sim.step(state.variant.config().dt); ctx.request_repaint(); }
    !back
}

impl eframe::App for LiziApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if let Some(ref mut state) = self.state {
            // Lazily initialize GlRenderer when entering simulation
            if state.gl_renderer.is_none() {
                if let Some(gl) = frame.gl() {
                    let renderer = GlRenderer::new(gl);
                    state.gl_renderer = Some(Arc::new(renderer));
                }
            }

            let gl_ref = frame.gl();
            let gl_opt: Option<&std::sync::Arc<eframe::glow::Context>> = gl_ref;
            if !render_simulation_panels(ctx, gl_opt, state) {
                self.state = None;
            } else if !state.paused {
                ctx.request_repaint();
            }
        } else {
            self.render_preset_selection(ctx);
        }
    }
}