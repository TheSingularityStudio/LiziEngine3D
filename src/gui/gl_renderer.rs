use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::sim::StateSnapshot;
use crate::gui::interaction::OrbitCamera;
use crate::visual::colors::heatmap_rgb;

const FAR_PLANE: f32 = 20.0;
const NEAR_PLANE: f32 = 0.1;
const HEATMAP_MAX_TEX_SIZE: usize = 128;
const DEFAULT_POINT_SIZE: f32 = 6.0;
const GRID_COUNT: usize = 8;
const AXIS_LENGTH: f32 = 0.3;
const GRID_COLOR: [f32; 4] = [0.3, 0.3, 0.3, 0.5];

pub struct GlRenderer {
    program_particle: eframe::glow::Program,
    program_line: eframe::glow::Program,
    program_quad: eframe::glow::Program,
    line_vao: eframe::glow::VertexArray,
    line_vbo: eframe::glow::Buffer,
    quad_vao: eframe::glow::VertexArray,
    quad_vbo: eframe::glow::Buffer,
    particle_vao: eframe::glow::VertexArray,
    heatmap_tex: eframe::glow::Texture,
    tex_nx: AtomicUsize,
    tex_ny: AtomicUsize,
    particle_vbo: eframe::glow::Buffer,
}

impl GlRenderer {
    pub fn new(gl: &eframe::glow::Context) -> Self {
        unsafe {
            use eframe::glow::HasContext;
            let program_particle =
                create_program(gl, include_str!("shaders/particle.vert"), include_str!("shaders/particle.frag"));
            let program_line =
                create_program(gl, include_str!("shaders/line.vert"), include_str!("shaders/line.frag"));
            let program_quad =
                create_program(gl, include_str!("shaders/quad.vert"), include_str!("shaders/quad.frag"));

            // ---- line VAO ----
            let line_vao = gl.create_vertex_array().unwrap();
            let line_vbo = gl.create_buffer().unwrap();
            gl.bind_vertex_array(Some(line_vao));
            gl.bind_buffer(eframe::glow::ARRAY_BUFFER, Some(line_vbo));
            gl.vertex_attrib_pointer_f32(0, 3, eframe::glow::FLOAT, false, 16, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 4, eframe::glow::FLOAT, false, 16, 12);
            gl.enable_vertex_attrib_array(1);

            // ---- quad VAO ----
            let quad_vao = gl.create_vertex_array().unwrap();
            let quad_vbo = gl.create_buffer().unwrap();
            gl.bind_vertex_array(Some(quad_vao));
            gl.bind_buffer(eframe::glow::ARRAY_BUFFER, Some(quad_vbo));
            gl.vertex_attrib_pointer_f32(0, 3, eframe::glow::FLOAT, false, 20, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 2, eframe::glow::FLOAT, false, 20, 12);
            gl.enable_vertex_attrib_array(1);

            // ---- particle VAO ----
            let particle_vao = gl.create_vertex_array().unwrap();
            let particle_vbo = gl.create_buffer().unwrap();
            gl.bind_vertex_array(Some(particle_vao));
            gl.bind_buffer(eframe::glow::ARRAY_BUFFER, Some(particle_vbo));
            gl.vertex_attrib_pointer_f32(0, 3, eframe::glow::FLOAT, false, 28, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 4, eframe::glow::FLOAT, false, 28, 12);
            gl.enable_vertex_attrib_array(1);

            // ---- heatmap texture ----
            let heatmap_tex = gl.create_texture().unwrap();
            gl.bind_texture(eframe::glow::TEXTURE_2D, Some(heatmap_tex));
            gl.tex_parameter_i32(eframe::glow::TEXTURE_2D, eframe::glow::TEXTURE_MIN_FILTER, eframe::glow::LINEAR as i32);
            gl.tex_parameter_i32(eframe::glow::TEXTURE_2D, eframe::glow::TEXTURE_MAG_FILTER, eframe::glow::LINEAR as i32);
            gl.tex_parameter_i32(eframe::glow::TEXTURE_2D, eframe::glow::TEXTURE_WRAP_S, eframe::glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(eframe::glow::TEXTURE_2D, eframe::glow::TEXTURE_WRAP_T, eframe::glow::CLAMP_TO_EDGE as i32);

            let init_data: Vec<u8> = vec![128; HEATMAP_MAX_TEX_SIZE * HEATMAP_MAX_TEX_SIZE * 4];
            gl.tex_image_2d(
                eframe::glow::TEXTURE_2D,
                0,
                eframe::glow::RGBA as i32,
                HEATMAP_MAX_TEX_SIZE as i32,
                HEATMAP_MAX_TEX_SIZE as i32,
                0,
                eframe::glow::RGBA,
                eframe::glow::UNSIGNED_BYTE,
                eframe::glow::PixelUnpackData::Slice(Some(&init_data)),
            );

            gl.bind_vertex_array(None);
            gl.bind_buffer(eframe::glow::ARRAY_BUFFER, None);

            Self {
                program_particle,
                program_line,
                program_quad,
                line_vao,
                line_vbo,
                quad_vao,
                quad_vbo,
                particle_vao,
                heatmap_tex,
                tex_nx: AtomicUsize::new(0),
                tex_ny: AtomicUsize::new(0),
                particle_vbo,
            }
        }
    }

    pub fn update_heatmap(
        &self,
        gl: &eframe::glow::Context,
        v: &ndarray::Array3<f64>,
        v_min: f64,
        v_max: f64,
    ) {
        use eframe::glow::HasContext;
        let (nx, ny, _) = v.dim();
        if nx == 0 || ny == 0 {
            return;
        }
        let mk = v.dim().2 / 2;
        let tnx = nx.min(HEATMAP_MAX_TEX_SIZE);
        let tny = ny.min(HEATMAP_MAX_TEX_SIZE);
        let si = if nx > tnx { nx / tnx } else { 1 };
        let sj = if ny > tny { ny / tny } else { 1 };
        let anx = (nx + si - 1) / si;
        let any = (ny + sj - 1) / sj;

        let mut px = Vec::with_capacity(anx * any * 4);
        for j in (0..ny).step_by(sj).rev() {
            for i in (0..nx).step_by(si) {
                let (r, g, b) = heatmap_rgb(v[[i, j, mk]], v_min, v_max);
                px.extend_from_slice(&[r, g, b, 255]);
            }
        }
        unsafe {
            gl.bind_texture(eframe::glow::TEXTURE_2D, Some(self.heatmap_tex));
            gl.tex_image_2d(
                eframe::glow::TEXTURE_2D, 0,
                eframe::glow::RGBA as i32,
                anx as i32, any as i32, 0,
                eframe::glow::RGBA,
                eframe::glow::UNSIGNED_BYTE,
                eframe::glow::PixelUnpackData::Slice(Some(&px)),
            );
            gl.bind_texture(eframe::glow::TEXTURE_2D, None);
        }
        self.tex_nx.store(anx, Ordering::Relaxed);
        self.tex_ny.store(any, Ordering::Relaxed);
    }

    pub fn render_scene(
        &self,
        gl: &eframe::glow::Context,
        snapshot: &StateSnapshot,
        camera: &OrbitCamera,
        show_heatmap: bool,
        show_grid: bool,
        show_axes: bool,
        screen_rect: (f32, f32, f32, f32),
    ) {
        unsafe {
            use eframe::glow::HasContext;
            let (x, y, w, h) = screen_rect;
            if w <= 0.0 || h <= 0.0 {
                return;
            }
            gl.viewport(x as i32, y as i32, w as i32, h as i32);
            gl.enable(eframe::glow::DEPTH_TEST);
            gl.enable(eframe::glow::BLEND);
            gl.blend_func(eframe::glow::SRC_ALPHA, eframe::glow::ONE_MINUS_SRC_ALPHA);

            let aspect = w / h;
            let mvp = build_mvp(camera, aspect);

            // ---- heatmap quad ----
            if show_heatmap && self.tex_nx.load(Ordering::Relaxed) > 0 && self.tex_ny.load(Ordering::Relaxed) > 0 {
                gl.use_program(Some(self.program_quad));
                gl.uniform_matrix_4_f32_slice(
                    gl.get_uniform_location(self.program_quad, "uMVP").as_ref(),
                    false,
                    &mvp,
                );
                gl.active_texture(eframe::glow::TEXTURE0);
                gl.bind_texture(eframe::glow::TEXTURE_2D, Some(self.heatmap_tex));
                gl.uniform_1_i32(
                    gl.get_uniform_location(self.program_quad, "uTexture").as_ref(),
                    0,
                );

                let qv: [f32; 30] = [
                    0.0, 0.0, 0.5, 0.0, 0.0,
                    1.0, 0.0, 0.5, 1.0, 0.0,
                    1.0, 1.0, 0.5, 1.0, 1.0,
                    0.0, 0.0, 0.5, 0.0, 0.0,
                    1.0, 1.0, 0.5, 1.0, 1.0,
                    0.0, 1.0, 0.5, 0.0, 1.0,
                ];
                gl.bind_buffer(eframe::glow::ARRAY_BUFFER, Some(self.quad_vbo));
                gl.buffer_data_u8_slice(
                    eframe::glow::ARRAY_BUFFER,
                    as_u8_slice(&qv),
                    eframe::glow::DYNAMIC_DRAW,
                );
                gl.bind_vertex_array(Some(self.quad_vao));
                gl.draw_arrays(eframe::glow::TRIANGLES, 0, 6);
                gl.bind_vertex_array(None);
            }

            // ---- grid ----
            if show_grid {
                gl.use_program(Some(self.program_line));
                gl.uniform_matrix_4_f32_slice(
                    gl.get_uniform_location(self.program_line, "uMVP").as_ref(),
                    false,
                    &mvp,
                );
                let mut lv: Vec<f32> = Vec::new();
                for i in 0..=GRID_COUNT {
                    let f = i as f32 / GRID_COUNT as f32;
                    lv.extend_from_slice(&[
                        f, 0.0, 0.0, GRID_COLOR[0], GRID_COLOR[1], GRID_COLOR[2], GRID_COLOR[3],
                        f, 1.0, 0.0, GRID_COLOR[0], GRID_COLOR[1], GRID_COLOR[2], GRID_COLOR[3],
                    ]);
                    lv.extend_from_slice(&[
                        0.0, f, 0.0, GRID_COLOR[0], GRID_COLOR[1], GRID_COLOR[2], GRID_COLOR[3],
                        1.0, f, 0.0, GRID_COLOR[0], GRID_COLOR[1], GRID_COLOR[2], GRID_COLOR[3],
                    ]);
                }
                gl.bind_buffer(eframe::glow::ARRAY_BUFFER, Some(self.line_vbo));
                gl.buffer_data_u8_slice(
                    eframe::glow::ARRAY_BUFFER,
                    as_u8_slice(&lv),
                    eframe::glow::DYNAMIC_DRAW,
                );
                gl.bind_vertex_array(Some(self.line_vao));
                gl.draw_arrays(eframe::glow::LINES, 0, (lv.len() / 7) as i32);
                gl.bind_vertex_array(None);
            }

            // ---- axes ----
            if show_axes {
                gl.use_program(Some(self.program_line));
                gl.uniform_matrix_4_f32_slice(
                    gl.get_uniform_location(self.program_line, "uMVP").as_ref(),
                    false,
                    &mvp,
                );
                let al = AXIS_LENGTH;
                let av: [f32; 42] = [
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                    al,  0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                    0.0, al,  0.0, 0.0, 1.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                    0.0, 0.0, al,  0.0, 0.0, 1.0, 1.0,
                ];
                gl.bind_buffer(eframe::glow::ARRAY_BUFFER, Some(self.line_vbo));
                gl.buffer_data_u8_slice(
                    eframe::glow::ARRAY_BUFFER,
                    as_u8_slice(&av),
                    eframe::glow::DYNAMIC_DRAW,
                );
                gl.bind_vertex_array(Some(self.line_vao));
                gl.draw_arrays(eframe::glow::LINES, 0, 6);
                gl.bind_vertex_array(None);
            }

            // ---- particles ----
            {
                gl.use_program(Some(self.program_particle));
                gl.uniform_matrix_4_f32_slice(
                    gl.get_uniform_location(self.program_particle, "uMVP").as_ref(),
                    false,
                    &mvp,
                );
                gl.uniform_1_f32(
                    gl.get_uniform_location(self.program_particle, "uPointSize").as_ref(),
                    DEFAULT_POINT_SIZE,
                );
                let n = snapshot.x.len();
                let mut pd: Vec<f32> = Vec::with_capacity(n * 7);
                for i in 0..n {
                    let (r, g, b) = if snapshot.q[i] < 0.0 {
                        (0.0, 1.0, 1.0)
                    } else {
                        (1.0, 1.0, 1.0)
                    };
                    pd.extend_from_slice(&[
                        (snapshot.x[i] / snapshot.lx) as f32,
                        (snapshot.y[i] / snapshot.ly) as f32,
                        (snapshot.z[i] / snapshot.lz) as f32,
                        r, g, b, 1.0,
                    ]);
                }
                gl.bind_buffer(eframe::glow::ARRAY_BUFFER, Some(self.particle_vbo));
                gl.buffer_data_u8_slice(
                    eframe::glow::ARRAY_BUFFER,
                    as_u8_slice(&pd),
                    eframe::glow::DYNAMIC_DRAW,
                );
                gl.bind_vertex_array(Some(self.particle_vao));
                gl.draw_arrays(eframe::glow::POINTS, 0, n as i32);
                gl.bind_vertex_array(None);
            }

            // ---- cleanup ----
            gl.disable(eframe::glow::BLEND);
            gl.disable(eframe::glow::DEPTH_TEST);
            gl.use_program(None);
            gl.bind_buffer(eframe::glow::ARRAY_BUFFER, None);
            gl.bind_vertex_array(None);
        }
    }
}

impl Drop for GlRenderer {
    fn drop(&mut self) {
        // We cannot drop OpenGL resources here safely because:
        // 1) we don't have a GL context at drop time,
        // 2) the native window may already be destroyed.
        // The resources are released when the GL context is destroyed.
    }
}

fn build_mvp(camera: &OrbitCamera, aspect: f32) -> [f32; 16] {
    let d = camera.distance.max(0.1);
    let (ca, sa) = camera.azimuth.sin_cos();
    let (ce, se) = camera.elevation.sin_cos();
    // View matrix: orbit camera look-at
    let v = [
        ca, sa * se, -sa * ce, 0.0,
        0.0, ce, se, 0.0,
        sa, -ca * se, ca * ce, 0.0,
        0.0, 0.0, -d, 1.0,
    ];
    // Model matrix: translate so (0.5,0.5,0.5) becomes origin
    let m = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -(camera.target_x - 0.5), -(camera.target_y - 0.5), -(camera.target_z - 0.5), 1.0,
    ];
    let fov = 45.0f32.to_radians();
    let ff = 1.0 / (fov * 0.5).tan();
    let p = [
        ff / aspect, 0.0, 0.0, 0.0,
        0.0, ff, 0.0, 0.0,
        0.0, 0.0, (FAR_PLANE + NEAR_PLANE) / (NEAR_PLANE - FAR_PLANE), -1.0,
        0.0, 0.0, 2.0 * FAR_PLANE * NEAR_PLANE / (NEAR_PLANE - FAR_PLANE), 0.0,
    ];
    mat4_mul(&p, &mat4_mul(&v, &m))
}

fn mat4_mul(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut r = [0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            r[i * 4 + j] = a[i * 4] * b[j]
                + a[i * 4 + 1] * b[4 + j]
                + a[i * 4 + 2] * b[8 + j]
                + a[i * 4 + 3] * b[12 + j];
        }
    }
    r
}

unsafe fn create_program(
    gl: &eframe::glow::Context,
    vs_src: &str,
    fs_src: &str,
) -> eframe::glow::Program {
    use eframe::glow::HasContext;
    let vs = gl.create_shader(eframe::glow::VERTEX_SHADER).unwrap();
    gl.shader_source(vs, vs_src);
    gl.compile_shader(vs);
    if !gl.get_shader_compile_status(vs) {
        panic!("VS compile error: {}", gl.get_shader_info_log(vs));
    }
    let fs = gl.create_shader(eframe::glow::FRAGMENT_SHADER).unwrap();
    gl.shader_source(fs, fs_src);
    gl.compile_shader(fs);
    if !gl.get_shader_compile_status(fs) {
        panic!("FS compile error: {}", gl.get_shader_info_log(fs));
    }
    let p = gl.create_program().unwrap();
    gl.attach_shader(p, vs);
    gl.attach_shader(p, fs);
    gl.link_program(p);
    if !gl.get_program_link_status(p) {
        panic!("Link error: {}", gl.get_program_info_log(p));
    }
    gl.delete_shader(vs);
    gl.delete_shader(fs);
    p
}

fn as_u8_slice(d: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            d.as_ptr() as *const u8,
            d.len() * std::mem::size_of::<f32>(),
        )
    }
}