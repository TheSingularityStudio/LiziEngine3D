use crate::core::sim::StateSnapshot;
use crate::gui::interaction::OrbitCamera;
use crate::visual::colors::heatmap_rgb;

pub struct GlRenderer {
    program_particle: eframe::glow::Program,
    program_line: eframe::glow::Program,
    program_quad: eframe::glow::Program,
    _line_vao: eframe::glow::VertexArray,
    line_vbo: eframe::glow::Buffer,
    _quad_vao: eframe::glow::VertexArray,
    quad_vbo: eframe::glow::Buffer,
    heatmap_tex: eframe::glow::Texture,
    _heatmap_dirty: bool,
    tex_nx: usize,
    tex_ny: usize,
    particle_vbo: eframe::glow::Buffer,
}

impl GlRenderer {
    pub fn new(gl: &eframe::glow::Context) -> Self {
        unsafe {
            use eframe::glow::HasContext;
            let program_particle = create_program(gl, include_str!("shaders/particle.vert"), include_str!("shaders/particle.frag"));
            let program_line = create_program(gl, include_str!("shaders/line.vert"), include_str!("shaders/line.frag"));
            let program_quad = create_program(gl, include_str!("shaders/quad.vert"), include_str!("shaders/quad.frag"));

            let line_vao = gl.create_vertex_array().unwrap();
            let line_vbo = gl.create_buffer().unwrap();
            gl.bind_vertex_array(Some(line_vao));
            gl.bind_buffer(eframe::glow::ARRAY_BUFFER, Some(line_vbo));
            gl.vertex_attrib_pointer_f32(0, 3, eframe::glow::FLOAT, false, 16, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 4, eframe::glow::FLOAT, false, 16, 12);
            gl.enable_vertex_attrib_array(1);

            let quad_vao = gl.create_vertex_array().unwrap();
            let quad_vbo = gl.create_buffer().unwrap();
            gl.bind_vertex_array(Some(quad_vao));
            gl.bind_buffer(eframe::glow::ARRAY_BUFFER, Some(quad_vbo));
            gl.vertex_attrib_pointer_f32(0, 3, eframe::glow::FLOAT, false, 20, 0);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(1, 2, eframe::glow::FLOAT, false, 20, 12);
            gl.enable_vertex_attrib_array(1);

            let particle_vbo = gl.create_buffer().unwrap();

            let heatmap_tex = gl.create_texture().unwrap();
            gl.bind_texture(eframe::glow::TEXTURE_2D, Some(heatmap_tex));
            gl.tex_parameter_i32(eframe::glow::TEXTURE_2D, eframe::glow::TEXTURE_MIN_FILTER, eframe::glow::LINEAR as i32);
            gl.tex_parameter_i32(eframe::glow::TEXTURE_2D, eframe::glow::TEXTURE_MAG_FILTER, eframe::glow::LINEAR as i32);
            gl.tex_parameter_i32(eframe::glow::TEXTURE_2D, eframe::glow::TEXTURE_WRAP_S, eframe::glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(eframe::glow::TEXTURE_2D, eframe::glow::TEXTURE_WRAP_T, eframe::glow::CLAMP_TO_EDGE as i32);

            let init_data: Vec<u8> = vec![128; 64 * 64 * 4];
            gl.tex_image_2d(eframe::glow::TEXTURE_2D, 0, eframe::glow::RGBA as i32, 64, 64, 0, eframe::glow::RGBA, eframe::glow::UNSIGNED_BYTE, eframe::glow::PixelUnpackData::Slice(Some(&init_data)));

            gl.bind_vertex_array(None);
            gl.bind_buffer(eframe::glow::ARRAY_BUFFER, None);
            Self { program_particle, program_line, program_quad, _line_vao: line_vao, line_vbo, _quad_vao: quad_vao, quad_vbo, heatmap_tex, _heatmap_dirty: true, tex_nx: 0, tex_ny: 0, particle_vbo }
        }
    }

    pub fn new_dummy() -> Self { panic!("GlRenderer requires OpenGL context") }

    pub fn update_heatmap(&mut self, gl: &eframe::glow::Context, v: &ndarray::Array3<f64>, v_min: f64, v_max: f64) {
        use eframe::glow::HasContext;
        let (nx, ny, _) = v.dim();
        if nx == 0 || ny == 0 { return; }
        let mk = v.dim().2 / 2;
        let tnx = nx.min(128); let tny = ny.min(128);
        let si = if nx > tnx { nx / tnx } else { 1 };
        let sj = if ny > tny { ny / tny } else { 1 };
        let anx = (nx + si - 1) / si; let any = (ny + sj - 1) / sj;

        let mut px = Vec::with_capacity(anx * any * 4);
        for j in (0..ny).step_by(sj).rev() {
            for i in (0..nx).step_by(si) {
                let (r, g, b) = heatmap_rgb(v[[i, j, mk]], v_min, v_max);
                px.extend_from_slice(&[r, g, b, 255]);
            }
        }
        unsafe {
            gl.bind_texture(eframe::glow::TEXTURE_2D, Some(self.heatmap_tex));
            gl.tex_image_2d(eframe::glow::TEXTURE_2D, 0, eframe::glow::RGBA as i32, anx as i32, any as i32, 0, eframe::glow::RGBA, eframe::glow::UNSIGNED_BYTE, eframe::glow::PixelUnpackData::Slice(Some(&px)));
            gl.bind_texture(eframe::glow::TEXTURE_2D, None);
        }
        self.tex_nx = anx; self.tex_ny = any;
    }

    pub fn render_scene(&self, gl: &eframe::glow::Context, snapshot: &StateSnapshot, camera: &OrbitCamera, show_heatmap: bool, show_grid: bool, show_axes: bool, screen_rect: (f32, f32, f32, f32)) {
        unsafe {
            use eframe::glow::HasContext;
            let (x, y, w, h) = screen_rect;
            if w <= 0.0 || h <= 0.0 { return; }
            gl.viewport(x as i32, y as i32, w as i32, h as i32);
            gl.enable(eframe::glow::DEPTH_TEST);
            gl.enable(eframe::glow::BLEND);
            gl.blend_func(eframe::glow::SRC_ALPHA, eframe::glow::ONE_MINUS_SRC_ALPHA);

            let aspect = w / h;
            let mvp = build_mvp(camera, aspect);

            if show_heatmap && self.tex_nx > 0 && self.tex_ny > 0 {
                gl.use_program(Some(self.program_quad));
                gl.uniform_matrix_4_f32_slice(gl.get_uniform_location(self.program_quad, "uMVP").as_ref(), false, &mvp);
                gl.active_texture(eframe::glow::TEXTURE0);
                gl.bind_texture(eframe::glow::TEXTURE_2D, Some(self.heatmap_tex));
                gl.uniform_1_i32(gl.get_uniform_location(self.program_quad, "uTexture").as_ref(), 0);
                let qv: [f32; 30] = [0.0,0.0,0.5,0.0,0.0, 1.0,0.0,0.5,1.0,0.0, 1.0,1.0,0.5,1.0,1.0, 0.0,0.0,0.5,0.0,0.0, 1.0,1.0,0.5,1.0,1.0, 0.0,1.0,0.5,0.0,1.0];
                gl.bind_buffer(eframe::glow::ARRAY_BUFFER, Some(self.quad_vbo));
                gl.buffer_data_u8_slice(eframe::glow::ARRAY_BUFFER, as_u8_slice(&qv), eframe::glow::DYNAMIC_DRAW);
                gl.bind_vertex_array(Some(self._quad_vao));
                gl.draw_arrays(eframe::glow::TRIANGLES, 0, 6);
                gl.bind_vertex_array(None);
            }

            if show_grid {
                gl.use_program(Some(self.program_line));
                gl.uniform_matrix_4_f32_slice(gl.get_uniform_location(self.program_line, "uMVP").as_ref(), false, &mvp);
                let mut lv: Vec<f32> = Vec::new();
                let gc = [0.3,0.3,0.3,0.5];
                for i in 0..=8usize {
                    let f = i as f32 / 8.0;
                    lv.extend_from_slice(&[f,0.0,0.0,gc[0],gc[1],gc[2],gc[3], f,1.0,0.0,gc[0],gc[1],gc[2],gc[3]]);
                    lv.extend_from_slice(&[0.0,f,0.0,gc[0],gc[1],gc[2],gc[3], 1.0,f,0.0,gc[0],gc[1],gc[2],gc[3]]);
                }
                gl.bind_buffer(eframe::glow::ARRAY_BUFFER, Some(self.line_vbo));
                gl.buffer_data_u8_slice(eframe::glow::ARRAY_BUFFER, as_u8_slice(&lv), eframe::glow::DYNAMIC_DRAW);
                gl.bind_vertex_array(Some(self._line_vao));
                gl.draw_arrays(eframe::glow::LINES, 0, (lv.len() / 4) as i32);
                gl.bind_vertex_array(None);
            }

            if show_axes {
                gl.use_program(Some(self.program_line));
                gl.uniform_matrix_4_f32_slice(gl.get_uniform_location(self.program_line, "uMVP").as_ref(), false, &mvp);
                let al = 0.3f32;
                let av: [f32; 42] = [
                    0.0,0.0,0.0,1.0,0.0,0.0,1.0, al,0.0,0.0,1.0,0.0,0.0,1.0,
                    0.0,0.0,0.0,0.0,1.0,0.0,1.0, 0.0,al,0.0,0.0,1.0,0.0,1.0,
                    0.0,0.0,0.0,0.0,0.0,1.0,1.0, 0.0,0.0,al,0.0,0.0,1.0,1.0,
                ];
                gl.bind_buffer(eframe::glow::ARRAY_BUFFER, Some(self.line_vbo));
                gl.buffer_data_u8_slice(eframe::glow::ARRAY_BUFFER, as_u8_slice(&av), eframe::glow::DYNAMIC_DRAW);
                gl.bind_vertex_array(Some(self._line_vao));
                gl.draw_arrays(eframe::glow::LINES, 0, 6);
                gl.bind_vertex_array(None);
            }

            {
                gl.use_program(Some(self.program_particle));
                gl.uniform_matrix_4_f32_slice(gl.get_uniform_location(self.program_particle, "uMVP").as_ref(), false, &mvp);
                gl.uniform_1_f32(gl.get_uniform_location(self.program_particle, "uPointSize").as_ref(), 6.0);
                let n = snapshot.x.len();
                let mut pd: Vec<f32> = Vec::with_capacity(n * 7);
                for i in 0..n {
                    let (r,g,b) = if snapshot.q[i] < 0.0 { (0.0,1.0,1.0) } else { (1.0,1.0,1.0) };
                    pd.extend_from_slice(&[(snapshot.x[i]/snapshot.lx) as f32, (snapshot.y[i]/snapshot.ly) as f32, (snapshot.z[i]/snapshot.lz) as f32, r, g, b, 1.0]);
                }
                gl.bind_buffer(eframe::glow::ARRAY_BUFFER, Some(self.particle_vbo));
                gl.buffer_data_u8_slice(eframe::glow::ARRAY_BUFFER, as_u8_slice(&pd), eframe::glow::DYNAMIC_DRAW);
                gl.vertex_attrib_pointer_f32(0, 3, eframe::glow::FLOAT, false, 28, 0);
                gl.enable_vertex_attrib_array(0);
                gl.vertex_attrib_pointer_f32(1, 4, eframe::glow::FLOAT, false, 28, 12);
                gl.enable_vertex_attrib_array(1);
                gl.draw_arrays(eframe::glow::POINTS, 0, n as i32);
                gl.disable_vertex_attrib_array(0);
                gl.disable_vertex_attrib_array(1);
            }

            gl.disable(eframe::glow::BLEND);
            gl.disable(eframe::glow::DEPTH_TEST);
            gl.use_program(None);
            gl.bind_buffer(eframe::glow::ARRAY_BUFFER, None);
            gl.bind_vertex_array(None);
        }
    }
}

impl Drop for GlRenderer { fn drop(&mut self) {} }

fn build_mvp(camera: &OrbitCamera, aspect: f32) -> [f32; 16] {
    let d = camera.distance.max(0.1);
    let (ca, sa) = camera.azimuth.sin_cos();
    let (ce, se) = camera.elevation.sin_cos();
    let v = [ca, sa*se, -sa*ce, 0.0, 0.0, ce, se, 0.0, sa, -ca*se, ca*ce, 0.0, 0.0, 0.0, -d, 1.0];
    let m = [1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, -(camera.target_x-0.5), -(camera.target_y-0.5), -(camera.target_z-0.5), 1.0];
    let f = 45.0f32.to_radians(); let near=0.1; let far=20.0;
    let ff = 1.0/(f*0.5).tan();
    let p = [ff/aspect,0.0,0.0,0.0, 0.0,ff,0.0,0.0, 0.0,0.0,(far+near)/(near-far),-1.0, 0.0,0.0,2.0*far*near/(near-far),0.0];
    mat4_mul(&p, &mat4_mul(&v, &m))
}

fn mat4_mul(a: &[f32;16], b: &[f32;16]) -> [f32;16] {
    let mut r=[0.0;16]; for i in 0..4 { for j in 0..4 { r[i*4+j]=a[i*4+0]*b[0*4+j]+a[i*4+1]*b[1*4+j]+a[i*4+2]*b[2*4+j]+a[i*4+3]*b[3*4+j]; } } r
}

unsafe fn create_program(gl: &eframe::glow::Context, vs_src: &str, fs_src: &str) -> eframe::glow::Program {
    use eframe::glow::HasContext;
    let vs = gl.create_shader(eframe::glow::VERTEX_SHADER).unwrap();
    gl.shader_source(vs, vs_src); gl.compile_shader(vs);
    if !gl.get_shader_compile_status(vs) { panic!("VS: {}", gl.get_shader_info_log(vs)); }
    let fs = gl.create_shader(eframe::glow::FRAGMENT_SHADER).unwrap();
    gl.shader_source(fs, fs_src); gl.compile_shader(fs);
    if !gl.get_shader_compile_status(fs) { panic!("FS: {}", gl.get_shader_info_log(fs)); }
    let p = gl.create_program().unwrap();
    gl.attach_shader(p, vs); gl.attach_shader(p, fs); gl.link_program(p);
    if !gl.get_program_link_status(p) { panic!("Link: {}", gl.get_program_info_log(p)); }
    gl.delete_shader(vs); gl.delete_shader(fs); p
}

fn as_u8_slice(d: &[f32]) -> &[u8] { unsafe { std::slice::from_raw_parts(d.as_ptr() as *const u8, d.len() * std::mem::size_of::<f32>()) } }
