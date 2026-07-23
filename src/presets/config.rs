use crate::core::boundary::BoundaryType;
use crate::core::grid::Grid3D;
use crate::core::particles::ParticleState;
use crate::core::sim::ElectrostaticSim3D;

#[derive(Debug, Clone)]
pub struct PresetConfig {
    pub name: &'static str,
    pub description: &'static str,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub eps: f64,
    pub dt: f64,
    pub seed: Option<u64>,
    pub boundary_type: BoundaryType,
    pub max_speed: Option<f64>,
    pub compute_fields_immediately: bool,
    pub particle_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetVariant {
    EmptyScene,
    SingleCharge,
    TwoChargesSame,
    TwoChargesOpposite,
    RandomParticles,
}

impl PresetVariant {
    pub fn all() -> &'static [PresetVariant] {
        &[PresetVariant::EmptyScene, PresetVariant::SingleCharge,
          PresetVariant::TwoChargesSame, PresetVariant::TwoChargesOpposite,
          PresetVariant::RandomParticles]
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            PresetVariant::EmptyScene => "空白场景",
            PresetVariant::SingleCharge => "单点电荷",
            PresetVariant::TwoChargesSame => "双电荷（同号）",
            PresetVariant::TwoChargesOpposite => "双电荷（异号）",
            PresetVariant::RandomParticles => "随机粒子",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            PresetVariant::EmptyScene => "完全空白的 3D 场景，无任何粒子，适合自行添加粒子进行实验。",
            PresetVariant::SingleCharge => "在网格中央放置一个单位正电荷，显示 3D 静电场 V 色图。",
            PresetVariant::TwoChargesSame => "在 x 轴两侧放置两个同号正电荷，观察叠加电场。",
            PresetVariant::TwoChargesOpposite => "在 x 轴两侧放置一正一负电荷，观察 3D 偶极子电场。",
            PresetVariant::RandomParticles => "500 个随机初始化的粒子在 3D 静电场中运动。",
        }
    }

    pub fn config(&self) -> PresetConfig {
        match self {
            PresetVariant::EmptyScene => PresetConfig {
                name: "空白场景", description: "",
                nx: 32, ny: 32, nz: 32, dx: 1.0, dy: 1.0, dz: 1.0,
                eps: 1e-12, dt: 0.05, seed: Some(0),
                boundary_type: BoundaryType::Periodic, max_speed: Some(10.0),
                compute_fields_immediately: false, particle_count: 0,
            },
            PresetVariant::SingleCharge => PresetConfig {
                name: "单点电荷", description: "",
                nx: 32, ny: 32, nz: 32, dx: 1.0, dy: 1.0, dz: 1.0,
                eps: 1e-12, dt: 0.05, seed: Some(0),
                boundary_type: BoundaryType::Periodic, max_speed: None,
                compute_fields_immediately: true, particle_count: 1,
            },
            PresetVariant::TwoChargesSame => PresetConfig {
                name: "双电荷（同号）", description: "",
                nx: 32, ny: 32, nz: 32, dx: 1.0, dy: 1.0, dz: 1.0,
                eps: 1e-12, dt: 0.05, seed: Some(0),
                boundary_type: BoundaryType::Periodic, max_speed: Some(10.0),
                compute_fields_immediately: false, particle_count: 2,
            },
            PresetVariant::TwoChargesOpposite => PresetConfig {
                name: "双电荷（异号）", description: "",
                nx: 32, ny: 32, nz: 32, dx: 1.0, dy: 1.0, dz: 1.0,
                eps: 1e-12, dt: 0.05, seed: Some(0),
                boundary_type: BoundaryType::Periodic, max_speed: Some(10.0),
                compute_fields_immediately: false, particle_count: 2,
            },
            PresetVariant::RandomParticles => PresetConfig {
                name: "随机粒子", description: "",
                nx: 32, ny: 32, nz: 32, dx: 1.0, dy: 1.0, dz: 1.0,
                eps: 1e-12, dt: 0.05, seed: Some(0),
                boundary_type: BoundaryType::Periodic, max_speed: Some(10.0),
                compute_fields_immediately: false, particle_count: 500,
            },
        }
    }

    pub fn build_particles(&self, grid: &Grid3D) -> ParticleState {
        match self {
            PresetVariant::EmptyScene => ParticleState::zeros(0, Some(0)),
            PresetVariant::SingleCharge => {
                let mut particles = ParticleState::zeros(1, Some(0));
                particles.x[0] = 0.5 * grid.lx();
                particles.y[0] = 0.5 * grid.ly();
                particles.z[0] = 0.5 * grid.lz();
                particles
            }
            PresetVariant::TwoChargesSame => {
                let charges = vec![1.0, 1.0];
                let mut particles = ParticleState::with_charges(2, Some(0), &charges);
                particles.x[0] = 0.25 * grid.lx();
                particles.y[0] = 0.5 * grid.ly();
                particles.z[0] = 0.5 * grid.lz();
                particles.x[1] = 0.75 * grid.lx();
                particles.y[1] = 0.5 * grid.ly();
                particles.z[1] = 0.5 * grid.lz();
                particles
            }
            PresetVariant::TwoChargesOpposite => {
                let charges = vec![1.0, -1.0];
                let mut particles = ParticleState::with_charges(2, Some(0), &charges);
                particles.x[0] = 0.25 * grid.lx();
                particles.y[0] = 0.5 * grid.ly();
                particles.z[0] = 0.5 * grid.lz();
                particles.x[1] = 0.75 * grid.lx();
                particles.y[1] = 0.5 * grid.ly();
                particles.z[1] = 0.5 * grid.lz();
                particles
            }
            PresetVariant::RandomParticles => {
                use rand::Rng;
                use rand::rngs::StdRng;
                use rand::SeedableRng;
                let n = 500;
                let mut particles = ParticleState::zeros(n, Some(0));
                let lx = grid.lx();
                let ly = grid.ly();
                let lz = grid.lz();
                let mut rng = StdRng::seed_from_u64(123);
                for i in 0..n {
                    particles.x[i] = rng.gen::<f64>() * lx;
                    particles.y[i] = rng.gen::<f64>() * ly;
                    particles.z[i] = rng.gen::<f64>() * lz;
                    particles.vx[i] = (rng.gen::<f64>() - 0.5) * 0.02;
                    particles.vy[i] = (rng.gen::<f64>() - 0.5) * 0.02;
                    particles.vz[i] = (rng.gen::<f64>() - 0.5) * 0.02;
                }
                particles
            }
        }
    }

    pub fn create_sim(&self) -> ElectrostaticSim3D {
        let config = self.config();
        let grid = Grid3D::new(config.nx, config.ny, config.nz, config.dx, config.dy, config.dz);
        let particles = self.build_particles(&grid);
        let mut sim = ElectrostaticSim3D::with_config(grid, particles, config.eps, config.boundary_type, config.max_speed);
        if config.compute_fields_immediately {
            sim.compute_fields();
        }
        sim
    }
}