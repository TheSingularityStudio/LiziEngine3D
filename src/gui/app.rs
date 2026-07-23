use std::sync::Arc;
use eframe::egui;
use egui::menu;

use crate::gui::interaction::{InteractionState, ToolMode, HoveredParticleInfo};
use crate::gui::gl_renderer::GlRenderer;
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
    show_about_dialog: bool,
    show_shortcuts_dialog: bool,
    message_dialog: Option<String>,
}

pub struct LiziApp {
    state: Option<SimulationState>,
    gl_renderer: Option<GlRenderer>,
    gl: Option<Arc<eframe::glow::Context>>,
}

impl Default for LiziApp {
    fn default() -> Self { Self { state: None, gl_renderer: None, gl: None } }
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
                let (gl_renderer, gl) = cc.gl.as_ref().map(|gl| {
                    let renderer = GlRenderer::new(gl.as_ref());
                    (renderer, Some(gl.clone()))
                }).unwrap_or((GlRenderer::new_dummy(), None));
                Ok(Box::new(LiziApp { state: None, gl_renderer: Some(gl_renderer), gl }))
            }),
        );
    }

    fn render_preset_selection(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(60.0);
                ui.heading("LiziEngine3D");
                ui.label("3D PIC Simulator (Glow Enhanced)");
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
                                    show_about_dialog: false, show_shortcuts_dialog: false,
                                    message_dialog: None,
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
                let mut sh = state.interaction.show_heatmap;
                if ui.checkbox(&mut sh, "Show Heatmap").changed() { state.interaction.show_heatmap = sh; }
                let mut sg = state.interaction.show_grid;
                if ui.checkbox(&mut sg, "Show Grid").changed() { state.interaction.show_grid = sg; }
                let mut sa = state.interaction.show_axes;
                if ui.checkbox(&mut sa, "Show Axes").changed() { state.interaction.show_axes = sa; }
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
                ui.label("Rust + egui + glow (OpenGL)");
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
                ui.label("  Drag — click & drag particles");
                ui.label("  Place — click to create particle");
                ui.label("  Delete — click to remove particle");
                ui.label("  Inspect — rotate/zoom/pan 3D view");
                ui.add_space(8.0);
                ui.heading("3D Controls"); ui.separator();
                ui.label("  Left drag — rotate");
                ui.label("  Scroll — zoom");
                ui.label("  Right drag — pan");
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

fn render_gl_canvas(
    ctx: &egui::Context, state: &mut SimulationState,
    gl_renderer: &mut GlRenderer, gl: &eframe::glow::Context,
) {
    let sim = &mut state.sim;
    let v_min = &mut state.v_min;
    let v_max = &mut state.v_max;
    let interaction = &mut state.interaction;

    egui::CentralPanel::default().show(ctx, |ui| {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click());
        ui.painter().rect_filled(rect, 0.0, ui.style().visuals.panel_fill);

        let screen_rect = ctx.input(|i| {
            let screen = i.screen_rect;
            let scale = i.pixels_per_point();
            let x = (rect.min.x * scale) as f32;
            let gl_y = ((screen.height() - rect.max.y) * scale) as f32;
            let w = (rect.width() * scale) as f32;
            let h = (rect.height() * scale) as f32;
            (x, gl_y, w, h)
        });

        let mouse_pos = ctx.input(|i| i.pointer.hover_pos());
        let mouse_down = ctx.input(|i| i.pointer.any_down());
        let mouse_primary = ctx.input(|i| i.pointer.button_down(egui::PointerButton::Primary));
        let mouse_secondary = ctx.input(|i| i.pointer.button_down(egui::PointerButton::Secondary));
        let scroll_delta = ctx.input(|i| i.raw_scroll_delta);

        if sim.v.is_none() || sim.ex.is_none() || sim.ey.is_none() || sim.ez.is_none() {
            sim.compute_fields();
        }
        let snapshot = sim.get_state_snapshot();

        let mut min_v = f64::MAX; let mut max_v = f64::MIN;
        for val in snapshot.v.iter() {
            if *val < min_v { min_v = *val; }
            if *val > max_v { max_v = *val; }
        }
        *v_min = *v_min * 0.9 + min_v * 0.1;
        *v_max = *v_max * 0.9 + max_v * 0.1;
        if (*v_max - *v_min).abs() < 1e-12 { *v_max = *v_min + 1.0; }

        gl_renderer.update_heatmap(gl, &snapshot.v, *v_min, *v_max);

        // Camera control (Inspect mode)
        if interaction.tool_mode == ToolMode::Inspect {
            if let Some(pos) = mouse_pos {
                if rect.contains(pos) {
                    if mouse_primary && !mouse_secondary {
                        if let Some(last) = interaction.camera.last_mouse_pos {
                            let dx = pos.x - last.0;
                            let dy = pos.y - last.1;
                            interaction.camera.azimuth += dx * 0.008;
                            interaction.camera.elevation = (interaction.camera.elevation + dy * 0.008).clamp(-1.5, 1.5);
                        }
                        interaction.camera.last_mouse_pos = Some((pos.x, pos.y));
                    } else if mouse_secondary {
                        if let Some(last) = interaction.camera.last_mouse_pos {
                            let dx = (pos.x - last.0) * 0.005 * interaction.camera.distance;
                            let dy = (pos.y - last.1) * 0.005 * interaction.camera.distance;
                            let ca = interaction.camera.azimuth.cos();
                            let sa = interaction.camera.azimuth.sin();
                            interaction.camera.target_x -= dx * ca;
                            interaction.camera.target_z -= dx * sa;
                            interaction.camera.target_y += dy;
                        }
                        interaction.camera.last_mouse_pos = Some((pos.x, pos.y));
                    } else {
                        interaction.camera.last_mouse_pos = None;
                    }
                    if scroll_delta.y != 0.0 {
                        let factor = if scroll_delta.y > 0.0 { 0.9 } else { 1.1 };
                        interaction.camera.distance = (interaction.camera.distance * factor).clamp(0.3, 15.0);
                    }
                }
            }

            // Hover detection
            interaction.hovered_particle = None;
            if !mouse_down {
                if let Some(pos) = mouse_pos {
                    if rect.contains(pos) {
                        let scale = ctx.input(|i| i.pixels_per_point());
                        let mx = (pos.x - rect.min.x) * scale;
                        let my = (rect.max.y - pos.y) * scale;
                        let aspect = rect.width() / rect.height().max(1.0);
                        let mvp = build_mvp_f32(&interaction.camera, aspect);
                        let hw = rect.width() * 0.5; let hh = rect.height() * 0.5;
                        let mut best = (interaction.selection_radius * rect.width() as f64).powi(2) as f32;
                        let mut best_i = None;
                        for i in 0..snapshot.x.len() {
                            let p: [f32; 4] = [(snapshot.x[i]/snapshot.lx) as f32, (snapshot.y[i]/snapshot.ly) as f32, (snapshot.z[i]/snapshot.lz) as f32, 1.0];
                            let pos2 = vec4_mul_mat4(&p, &mvp);
                            if pos2[2] < 0.0 { continue; }
                            let sx = pos2[0]/pos2[2]*hw + hw;
                            let sy = pos2[1]/pos2[2]*hh + hh;
                            let ddx = mx - sx; let ddy = (rect.height() - sy) - my;
                            let d2 = ddx*ddx + ddy*ddy;
                            if d2 < best { best = d2; best_i = Some(i); }
                        }
                        if let Some(idx) = best_i {
                            interaction.hovered_particle = Some(HoveredParticleInfo {
                                index: idx, x: snapshot.x[idx], y: snapshot.y[idx], z: snapshot.z[idx],
                                vx: snapshot.vx[idx], vy: snapshot.vy[idx], vz: snapshot.vz[idx],
                                q: snapshot.q[idx], m: snapshot.m[idx],
                            });
                        }
                    }
                }
            }
        }

        gl_renderer.render_scene(gl, &snapshot, &interaction.camera,
            interaction.show_heatmap, interaction.show_grid, interaction.show_axes, screen_rect);

        // Interaction
        handle_3d_interaction(interaction, sim, &snapshot, mouse_pos, mouse_down, mouse_primary, rect, &response, ctx);
    });
}

fn handle_3d_interaction(
    interaction: &mut InteractionState, sim: &mut ElectrostaticSim3D,
    snapshot: &crate::core::sim::StateSnapshot,
    mouse_pos: Option<egui::Pos2>, mouse_down: bool, mouse_primary: bool,
    rect: egui::Rect, _response: &egui::Response, ctx: &egui::Context,
) {
    let Some(pos) = mouse_pos else { return };
    if !rect.contains(pos) { return; }
    let nx = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0) as f64;
    let ny = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0) as f64;
    let wx = nx * snapshot.lx;
    let wy = (1.0 - ny) * snapshot.ly;

    if interaction.tool_mode == ToolMode::DragParticle && mouse_down && mouse_primary {
        if !interaction.dragging {
            let mut best = interaction.selection_radius * rect.width() as f64;
            let mut bi = None;
            for i in 0..snapshot.x.len() {
                let d = (((snapshot.x[i]/snapshot.lx)-nx).powi(2) + ((snapshot.y[i]/snapshot.ly)-(1.0-ny)).powi(2)).sqrt() * rect.width() as f64;
                if d < best { best = d; bi = Some(i); }
            }
            if let Some(idx) = bi { interaction.dragging = true; interaction.dragged_particle_index = Some(idx); }
        }
        if let Some(idx) = interaction.dragged_particle_index {
            sim.particles.x[idx] = wx.clamp(0.0, snapshot.lx);
            sim.particles.y[idx] = wy.clamp(0.0, snapshot.ly);
            sim.particles.vx[idx] = 0.0; sim.particles.vy[idx] = 0.0;
            sim.v = None; sim.ex = None; sim.ey = None; sim.ez = None;
        }
    }

    if interaction.tool_mode == ToolMode::PlaceParticle && ctx.input(|i| i.pointer.any_click()) {
        let q = interaction.place_params.charge;
        let m = interaction.place_params.mass;
        sim.particles.add_particle(wx.clamp(0.0, snapshot.lx), wy.clamp(0.0, snapshot.ly), 0.5*snapshot.lz, q, m, 0.0, 0.0, 0.0);
        sim.v = None; sim.ex = None; sim.ey = None; sim.ez = None;
    }

    if interaction.tool_mode == ToolMode::DeleteParticle && ctx.input(|i| i.pointer.any_click()) {
        let mut best = interaction.selection_radius * rect.width() as f64;
        let mut bi = None;
        for i in 0..snapshot.x.len() {
            let d = (((snapshot.x[i]/snapshot.lx)-nx).powi(2) + ((snapshot.y[i]/snapshot.ly)-(1.0-ny)).powi(2)).sqrt() * rect.width() as f64;
            if d < best { best = d; bi = Some(i); }
        }
        if let Some(idx) = bi { sim.particles.remove_particle(idx); sim.v = None; sim.ex = None; sim.ey = None; sim.ez = None; }
    }

    if !mouse_down { interaction.dragging = false; interaction.dragged_particle_index = None; }
}

fn build_mvp_f32(camera: &crate::gui::interaction::OrbitCamera, aspect: f32) -> [f32; 16] {
    let d = camera.distance.max(0.1);
    let (ca, sa) = camera.azimuth.sin_cos();
    let (ce, se) = camera.elevation.sin_cos();
    let tx = camera.target_x - 0.5;
    let ty = camera.target_y - 0.5;
    let tz = camera.target_z - 0.5;
    let view = [ca, sa*se, -sa*ce, 0.0, 0.0, ce, se, 0.0, sa, -ca*se, ca*ce, 0.0, 0.0, 0.0, -d, 1.0];
    let model = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -tx, -ty, -tz, 1.0];
    let f = 45.0f32.to_radians(); let near = 0.1; let far = 20.0;
    let ff = 1.0 / (f*0.5).tan();
    let proj = [ff/aspect, 0.0, 0.0, 0.0, 0.0, ff, 0.0, 0.0, 0.0, 0.0, (far+near)/(near-far), -1.0, 0.0, 0.0, 2.0*far*near/(near-far), 0.0];
    let vm = mat4_mul_f32(&view, &model);
    mat4_mul_f32(&proj, &vm)
}

fn mat4_mul_f32(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut r = [0.0f32; 16];
    for i in 0..4 { for j in 0..4 { r[i*4+j] = a[i*4+0]*b[0*4+j] + a[i*4+1]*b[1*4+j] + a[i*4+2]*b[2*4+j] + a[i*4+3]*b[3*4+j]; } }
    r
}

fn vec4_mul_mat4(v: &[f32; 4], m: &[f32; 16]) -> [f32; 4] {
    [v[0]*m[0]+v[1]*m[4]+v[2]*m[8]+v[3]*m[12], v[0]*m[1]+v[1]*m[5]+v[2]*m[9]+v[3]*m[13], v[0]*m[2]+v[1]*m[6]+v[2]*m[10]+v[3]*m[14], v[0]*m[3]+v[1]*m[7]+v[2]*m[11]+v[3]*m[15]]
}

fn render_simulation_panels(ctx: &egui::Context, state: &mut SimulationState, gl_renderer: &mut GlRenderer, gl: &eframe::glow::Context) -> bool {
    let back = render_menu_bar(ctx, state);
    render_dialogs(ctx, state);
    render_left_panel(ctx, state);
    render_right_panel(ctx, state);
    render_gl_canvas(ctx, state, gl_renderer, gl);
    if !state.paused { state.sim.step(state.variant.config().dt); ctx.request_repaint(); }
    !back
}

impl eframe::App for LiziApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(ref mut state) = self.state {
            if let (Some(ref mut renderer), Some(ref gl)) = (&mut self.gl_renderer, &self.gl) {
                if !render_simulation_panels(ctx, state, renderer, gl.as_ref()) {
                    self.state = None;
                } else if !state.paused { ctx.request_repaint(); }
            }
        } else { self.render_preset_selection(ctx); }
    }
}