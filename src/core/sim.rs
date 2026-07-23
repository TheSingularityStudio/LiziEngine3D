use ndarray::Array3;
use crate::core::grid::Grid3D;
use crate::core::particles::ParticleState;
use crate::core::scatter::scatter_charges_to_grid;
use crate::core::poisson_solver::{PoissonSolver, compute_e_from_potential_periodic};
use crate::core::interp::gather_field_to_particles_trilinear;
use crate::core::integrator::step_half_implicit_euler;
use crate::core::boundary::{BoundaryType, apply_boundary_conditions, apply_speed_limit};

/// 3D 静电（电场-粒子）CPU 模拟器（PIC 风格实现；单位电荷、单位质量）
#[derive(Debug)]
pub struct ElectrostaticSim3D {
    pub grid: Grid3D,
    pub particles: ParticleState,
    pub eps_poisson: f64,
    pub rho: Option<Array3<f64>>,
    pub v: Option<Array3<f64>>,
    pub ex: Option<Array3<f64>>,
    pub ey: Option<Array3<f64>>,
    pub ez: Option<Array3<f64>>,
    pub boundary_type: BoundaryType,
    pub max_speed: Option<f64>,
    pub gravity_enabled: bool,
    pub gravity_x: f64,
    pub gravity_y: f64,
    pub gravity_z: f64,
    pub friction_enabled: bool,
    pub friction_damping: f64,
    poisson_solver: Option<PoissonSolver>,
}

impl ElectrostaticSim3D {
    pub fn new(grid: Grid3D, particles: ParticleState, eps_poisson: f64) -> Self {
        Self {
            poisson_solver: None,
            grid, particles, eps_poisson,
            rho: None, v: None, ex: None, ey: None, ez: None,
            boundary_type: BoundaryType::Periodic,
            max_speed: Some(10.0),
            gravity_enabled: false,
            gravity_x: 0.0, gravity_y: 0.0, gravity_z: -9.8,
            friction_enabled: false,
            friction_damping: 0.1,
        }
    }

    pub fn with_config(
        grid: Grid3D, particles: ParticleState, eps_poisson: f64,
        boundary_type: BoundaryType, max_speed: Option<f64>,
    ) -> Self {
        Self { poisson_solver: None, grid, particles, eps_poisson,
            rho: None, v: None, ex: None, ey: None, ez: None,
            boundary_type, max_speed,
            gravity_enabled: false, gravity_x: 0.0, gravity_y: 0.0, gravity_z: -9.8,
            friction_enabled: false, friction_damping: 0.1,
        }
    }

    fn get_or_init_solver(&mut self) -> &PoissonSolver {
        self.poisson_solver.get_or_insert_with(|| {
            PoissonSolver::new(self.grid.nx, self.grid.ny, self.grid.nz,
                self.grid.dx, self.grid.dy, self.grid.dz, self.eps_poisson)
        })
    }

    pub fn compute_fields(&mut self) {
        let rho = scatter_charges_to_grid(&self.grid, &self.particles);
        let eps = self.eps_poisson;
        let solver = self.get_or_init_solver();
        let v = solver.solve(&rho, eps);
        let (ex, ey, ez) = compute_e_from_potential_periodic(&v, self.grid.dx, self.grid.dy, self.grid.dz);

        let (fx, fy, fz) = gather_field_to_particles_trilinear(
            &self.grid, &self.particles, &ex, &ey, &ez);
        self.particles.fx = &fx * &self.particles.q;
        self.particles.fy = &fy * &self.particles.q;
        self.particles.fz = &fz * &self.particles.q;

        self.rho = Some(rho);
        self.v = Some(v);
        self.ex = Some(ex);
        self.ey = Some(ey);
        self.ez = Some(ez);
    }

    pub fn step(&mut self, dt: f64) {
        self.compute_fields();

        if self.gravity_enabled {
            for i in 0..self.particles.len() {
                self.particles.fx[i] += self.particles.m[i] * self.gravity_x;
                self.particles.fy[i] += self.particles.m[i] * self.gravity_y;
                self.particles.fz[i] += self.particles.m[i] * self.gravity_z;
            }
        }

        if self.friction_enabled && self.friction_damping > 0.0 {
            for i in 0..self.particles.len() {
                self.particles.fx[i] -= self.friction_damping * self.particles.vx[i];
                self.particles.fy[i] -= self.friction_damping * self.particles.vy[i];
                self.particles.fz[i] -= self.friction_damping * self.particles.vz[i];
            }
        }

        step_half_implicit_euler(&self.grid, &mut self.particles, dt);
        apply_boundary_conditions(&mut self.particles, &self.grid, self.boundary_type);
        if let Some(max_speed) = self.max_speed {
            apply_speed_limit(&mut self.particles, max_speed);
        }
    }

    pub fn set_particle_position(&mut self, index: usize, x: f64, y: f64, z: f64) {
        if index < self.particles.x.len() {
            self.particles.x[index] = x;
            self.particles.y[index] = y;
            self.particles.z[index] = z;
            self.particles.vx[index] = 0.0;
            self.particles.vy[index] = 0.0;
            self.particles.vz[index] = 0.0;
        }
    }

    pub fn get_state_snapshot(&mut self) -> StateSnapshot {
        if self.v.is_none() || self.ex.is_none() || self.ey.is_none() || self.ez.is_none() {
            self.compute_fields();
        }
        StateSnapshot {
            x: self.particles.x.clone(),
            y: self.particles.y.clone(),
            z: self.particles.z.clone(),
            vx: self.particles.vx.clone(),
            vy: self.particles.vy.clone(),
            vz: self.particles.vz.clone(),
            q: self.particles.q.clone(),
            m: self.particles.m.clone(),
            v: self.v.as_ref().expect("v should be set").clone(),
            ex: self.ex.as_ref().expect("ex should be set").clone(),
            ey: self.ey.as_ref().expect("ey should be set").clone(),
            ez: self.ez.as_ref().expect("ez should be set").clone(),
            lx: self.grid.lx(),
            ly: self.grid.ly(),
            lz: self.grid.lz(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub x: ndarray::Array1<f64>,
    pub y: ndarray::Array1<f64>,
    pub z: ndarray::Array1<f64>,
    pub vx: ndarray::Array1<f64>,
    pub vy: ndarray::Array1<f64>,
    pub vz: ndarray::Array1<f64>,
    pub q: ndarray::Array1<f64>,
    pub m: ndarray::Array1<f64>,
    pub v: Array3<f64>,
    pub ex: Array3<f64>,
    pub ey: Array3<f64>,
    pub ez: Array3<f64>,
    pub lx: f64,
    pub ly: f64,
    pub lz: f64,
}