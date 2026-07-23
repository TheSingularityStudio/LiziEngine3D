use crate::core::boundary::BoundaryType;
use crate::core::grid::Grid2D;
use crate::core::particles::ParticleState;
use crate::core::sim::ElectrostaticSim2D;

/// 预设场景配置
#[derive(Debug, Clone)]
pub struct PresetConfig {
    pub name: &'static str,
    pub description: &'static str,
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub eps: f64,
    pub dt: f64,
    pub seed: Option<u64>,
    pub boundary_type: BoundaryType,
    pub max_speed: Option<f64>,
    /// 是否在加载后立即计算场（单电荷等静态场景）
    pub compute_fields_immediately: bool,
    /// 粒子数量（用于 create_sim 构建粒子）
    pub particle_count: usize,
}

/// 预设变体枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetVariant {
    EmptyScene,
    SingleCharge,
    TwoChargesSame,
    TwoChargesOpposite,
    RandomParticles,
}

impl PresetVariant {
    /// 获取所有预设变体列表
    pub fn all() -> &'static [PresetVariant] {
        &[
            PresetVariant::EmptyScene,
            PresetVariant::SingleCharge,
            PresetVariant::TwoChargesSame,
            PresetVariant::TwoChargesOpposite,
            PresetVariant::RandomParticles,
        ]
    }

    /// 获取预设的显示名称
    pub fn display_name(&self) -> &'static str {
        match self {
            PresetVariant::EmptyScene => "空白场景",
            PresetVariant::SingleCharge => "单点电荷",
            PresetVariant::TwoChargesSame => "双电荷（同号）",
            PresetVariant::TwoChargesOpposite => "双电荷（异号）",
            PresetVariant::RandomParticles => "随机粒子",
        }
    }

    /// 获取预设的详细描述
    pub fn description(&self) -> &'static str {
        match self {
            PresetVariant::EmptyScene => {
                "完全空白的场景，无任何粒子，适合自行添加粒子进行实验。"
            }
            PresetVariant::SingleCharge => {
                "在网格中央放置一个单位正电荷，显示静电场 V 热力图。"
            }
            PresetVariant::TwoChargesSame => {
                "在网格左右两侧放置两个同号正电荷，观察叠加电场。"
            }
            PresetVariant::TwoChargesOpposite => {
                "在网格左右两侧放置一正一负电荷，观察偶极子电场。"
            }
            PresetVariant::RandomParticles => {
                "200 个随机初始化的粒子在静电场中运动，展示 PIC 模拟动画。"
            }
        }
    }

    /// 获取预设的完整配置
    pub fn config(&self) -> PresetConfig {
        match self {
            PresetVariant::EmptyScene => PresetConfig {
                name: "空白场景",
                description: "完全空白的场景，无任何粒子，适合自行添加粒子进行实验。",
                nx: 64,
                ny: 64,
                dx: 1.0,
                dy: 1.0,
                eps: 1e-12,
                dt: 0.05,
                seed: Some(0),
                boundary_type: BoundaryType::Periodic,
                max_speed: Some(10.0),
                compute_fields_immediately: false,
                particle_count: 0,
            },
            PresetVariant::SingleCharge => PresetConfig {
                name: "单点电荷",
                description: "在网格中央放置一个单位正电荷，显示静电场 V 热力图。",
                nx: 64,
                ny: 64,
                dx: 1.0,
                dy: 1.0,
                eps: 1e-12,
                dt: 0.05,
                seed: Some(0),
                boundary_type: BoundaryType::Periodic,
                max_speed: None,
                compute_fields_immediately: true,
                particle_count: 1,
            },
            PresetVariant::TwoChargesSame => PresetConfig {
                name: "双电荷（同号）",
                description: "在网格左右两侧放置两个同号正电荷，观察叠加电场。",
                nx: 64,
                ny: 64,
                dx: 1.0,
                dy: 1.0,
                eps: 1e-12,
                dt: 0.05,
                seed: Some(0),
                boundary_type: BoundaryType::Periodic,
                max_speed: Some(10.0),
                compute_fields_immediately: false,
                particle_count: 2,
            },
            PresetVariant::TwoChargesOpposite => PresetConfig {
                name: "双电荷（异号）",
                description: "在网格左右两侧放置一正一负电荷，观察偶极子电场。",
                nx: 64,
                ny: 64,
                dx: 1.0,
                dy: 1.0,
                eps: 1e-12,
                dt: 0.05,
                seed: Some(0),
                boundary_type: BoundaryType::Periodic,
                max_speed: Some(10.0),
                compute_fields_immediately: false,
                particle_count: 2,
            },
            PresetVariant::RandomParticles => PresetConfig {
                name: "随机粒子",
                description: "200 个随机初始化的粒子在静电场中运动，展示 PIC 模拟动画。",
                nx: 64,
                ny: 64,
                dx: 1.0,
                dy: 1.0,
                eps: 1e-12,
                dt: 0.05,
                seed: Some(0),
                boundary_type: BoundaryType::Periodic,
                max_speed: Some(10.0),
                compute_fields_immediately: false,
                particle_count: 200,
            },
        }
    }

    /// 根据预设配置构建粒子状态
    pub fn build_particles(&self, grid: &Grid2D) -> ParticleState {
        match self {
            PresetVariant::EmptyScene => {
                ParticleState::zeros(0, Some(0))
            }
            PresetVariant::SingleCharge => {
                let mut particles = ParticleState::zeros(1, Some(0));
                particles.x[0] = 0.5 * grid.lx();
                particles.y[0] = 0.5 * grid.ly();
                particles
            }
            PresetVariant::TwoChargesSame => {
                let charges = vec![1.0, 1.0];
                let mut particles = ParticleState::with_charges(2, Some(0), &charges);
                particles.x[0] = 0.25 * grid.lx();
                particles.y[0] = 0.5 * grid.ly();
                particles.x[1] = 0.75 * grid.lx();
                particles.y[1] = 0.5 * grid.ly();
                particles
            }
            PresetVariant::TwoChargesOpposite => {
                let charges = vec![1.0, -1.0];
                let mut particles = ParticleState::with_charges(2, Some(0), &charges);
                particles.x[0] = 0.25 * grid.lx();
                particles.y[0] = 0.5 * grid.ly();
                particles.x[1] = 0.75 * grid.lx();
                particles.y[1] = 0.5 * grid.ly();
                particles
            }
            PresetVariant::RandomParticles => {
                use rand::Rng;
                use rand::rngs::StdRng;
                use rand::SeedableRng;
                let n = 200;
                let mut particles = ParticleState::zeros(n, Some(0));
                let lx = grid.lx();
                let ly = grid.ly();
                let seed = 0u64.wrapping_add(123);
                let mut rng = StdRng::seed_from_u64(seed);
                for i in 0..n {
                    particles.x[i] = rng.gen::<f64>() * lx;
                    particles.y[i] = rng.gen::<f64>() * ly;
                    particles.vx[i] = (rng.gen::<f64>() - 0.5) * 0.02;
                    particles.vy[i] = (rng.gen::<f64>() - 0.5) * 0.02;
                }
                particles
            }
        }
    }

    /// 根据预设配置创建模拟器实例
    pub fn create_sim(&self) -> ElectrostaticSim2D {
        let config = self.config();
        let grid = Grid2D::new(config.nx, config.ny, config.dx, config.dy);
        let particles = self.build_particles(&grid);
        let mut sim = ElectrostaticSim2D::with_config(
            grid,
            particles,
            config.eps,
            config.boundary_type,
            config.max_speed,
        );
        if config.compute_fields_immediately {
            sim.compute_fields();
        }
        sim
    }
}