use ndarray::Array2;
use ndarray::Array3;
use crate::core::grid::Grid2D;
use crate::core::particles::ParticleState;
use crate::core::scatter::scatter_unit_charges_to_grid;
use crate::core::poisson_solver::{PoissonSolver, compute_e_from_potential_periodic};
use crate::core::interp::gather_field_to_particles_bilinear;
use crate::core::integrator::step_half_implicit_euler;
use crate::core::boundary::{BoundaryType, apply_boundary_conditions, apply_speed_limit};

/// 2D 静电（电场-粒子）CPU 模拟器（PIC 风格实现；单位电荷、单位质量）
///
/// 每个时间步的计算流程：
///   1) 将粒子散射到网格上，得到离散电荷密度 rho
///   2) 在周期边界条件下求解 Poisson：通过 FFT 求得电势 V
///   3) 在网格上计算电场：E = -∇V
///   4) 将网格电场通过 gather 插值回粒子位置，得到粒子受力（由于 q=1，故 F = E）
///   5) 使用半隐式欧拉对粒子积分更新（速度/位置）
#[derive(Debug)]
pub struct ElectrostaticSim2D {
    pub grid: Grid2D,
    pub particles: ParticleState,
    pub eps_poisson: f64,
    pub rho: Option<Array2<f64>>,
    pub v: Option<Array2<f64>>,
    pub ex: Option<Array2<f64>>,
    pub ey: Option<Array2<f64>>,
    /// 边界类型（周期或反弹）
    pub boundary_type: BoundaryType,
    /// 最高速度限制（None 表示无限制）
    pub max_speed: Option<f64>,
    /// 重力开关
    pub gravity_enabled: bool,
    /// X 方向重力加速度
    pub gravity_x: f64,
    /// Y 方向重力加速度
    pub gravity_y: f64,
    /// 摩擦力开关
    pub friction_enabled: bool,
    /// 摩擦阻尼系数
    pub friction_damping: f64,
    /// 缓存的 Poisson 求解器（带 FFT Handler 预分配）
    poisson_solver: Option<PoissonSolver>,
}

impl ElectrostaticSim2D {
    /// 创建新的模拟器实例（使用默认配置：周期边界，最高速度10.0）
    pub fn new(grid: Grid2D, particles: ParticleState, eps_poisson: f64) -> Self {
        Self {
            poisson_solver: None, // 首次使用时惰性初始化
            grid,
            particles,
            eps_poisson,
            rho: None,
            v: None,
            ex: None,
            ey: None,
            boundary_type: BoundaryType::Periodic,
            max_speed: Some(10.0),
            gravity_enabled: false,
            gravity_x: 0.0,
            gravity_y: -9.8,
            friction_enabled: false,
            friction_damping: 0.1,
        }
    }

    /// 创建自定义配置的模拟器实例
    pub fn with_config(
        grid: Grid2D,
        particles: ParticleState,
        eps_poisson: f64,
        boundary_type: BoundaryType,
        max_speed: Option<f64>,
    ) -> Self {
        Self {
            poisson_solver: None,
            grid,
            particles,
            eps_poisson,
            rho: None,
            v: None,
            ex: None,
            ey: None,
            boundary_type,
            max_speed,
            gravity_enabled: false,
            gravity_x: 0.0,
            gravity_y: -9.8,
            friction_enabled: false,
            friction_damping: 0.1,
        }
    }

    /// 获取或初始化 Poisson 求解器（使用缓存的 FFT Handler）
    fn get_or_init_solver(&mut self) -> &PoissonSolver {
        self.poisson_solver.get_or_insert_with(|| {
            PoissonSolver::new(
                self.grid.nx,
                self.grid.ny,
                self.grid.dx,
                self.grid.dy,
                self.eps_poisson,
            )
        })
    }

    /// 计算当前粒子状态下的全场：rho → V → Ex, Ey，并将电场 gather 到粒子
    pub fn compute_fields(&mut self) {
        let rho = scatter_unit_charges_to_grid(&self.grid, &self.particles);
        let eps = self.eps_poisson;
        let solver = self.get_or_init_solver();
        let v = solver.solve(&rho, eps);
        let (ex, ey) = compute_e_from_potential_periodic(&v, self.grid.dx, self.grid.dy);

        // gather 电场到粒子受力
        let (ex_at_parts, ey_at_parts) = gather_field_to_particles_bilinear(
            &self.grid,
            &self.particles,
            &ex,
            &ey,
        );
        // F = q * E
        self.particles.fx = &ex_at_parts * &self.particles.q;
        self.particles.fy = &ey_at_parts * &self.particles.q;

        self.rho = Some(rho);
        self.v = Some(v);
        self.ex = Some(ex);
        self.ey = Some(ey);
    }

    /// 执行一个时间步：计算场 + 积分 + 边界处理 + 速度限制
    pub fn step(&mut self, dt: f64) {
        self.compute_fields();
        
        // 应用重力（在积分前将重力叠加到受力上）
        if self.gravity_enabled {
            for i in 0..self.particles.len() {
                // F = m * g
                self.particles.fx[i] += self.particles.m[i] * self.gravity_x;
                self.particles.fy[i] += self.particles.m[i] * self.gravity_y;
            }
        }

        // 应用摩擦力（在积分前将阻尼力叠加到受力上：F = -damping * v）
        if self.friction_enabled && self.friction_damping > 0.0 {
            for i in 0..self.particles.len() {
                self.particles.fx[i] -= self.friction_damping * self.particles.vx[i];
                self.particles.fy[i] -= self.friction_damping * self.particles.vy[i];
            }
        }
        
        step_half_implicit_euler(&self.grid, &mut self.particles, dt);
        
        // 应用边界条件
        apply_boundary_conditions(&mut self.particles, &self.grid, self.boundary_type);
        
        // 应用最高速度限制
        if let Some(max_speed) = self.max_speed {
            apply_speed_limit(&mut self.particles, max_speed);
        }
    }

    /// 运行指定步数的模拟，按 record_every 间隔记录粒子位置快照
    ///
    /// 返回：positions, shape = (frames, N, 2)
    pub fn run(&mut self, dt: f64, steps: usize, record_every: usize) -> Array3<f64> {
        let n = self.particles.len();
        let num_frames = steps / record_every + if steps % record_every > 0 { 1 } else { 0 };
        let mut frames = Array3::zeros((num_frames, n, 2));
        let mut frame_idx = 0;

        for s in 0..steps {
            self.step(dt);
            if s % record_every == 0 {
                for p in 0..n {
                    frames[[frame_idx, p, 0]] = self.particles.x[p];
                    frames[[frame_idx, p, 1]] = self.particles.y[p];
                }
                frame_idx += 1;
            }
        }

        frames
    }

    /// 更新单个粒子的位置
    pub fn set_particle_position(&mut self, index: usize, x: f64, y: f64) {
        if index < self.particles.x.len() {
            self.particles.x[index] = x;
            self.particles.y[index] = y;
            // 重置速度
            self.particles.vx[index] = 0.0;
            self.particles.vy[index] = 0.0;
        }
    }

    /// 获取当前仿真状态快照（用于可视化/调试）
    pub fn get_state_snapshot(&mut self) -> StateSnapshot {
        if self.v.is_none() || self.ex.is_none() || self.ey.is_none() {
            self.compute_fields();
        }

        StateSnapshot {
            x: self.particles.x.clone(),
            y: self.particles.y.clone(),
            vx: self.particles.vx.clone(),
            vy: self.particles.vy.clone(),
            q: self.particles.q.clone(),
            v: self.v.as_ref().expect("v should be set after compute_fields").clone(),
            ex: self.ex.as_ref().expect("ex should be set after compute_fields").clone(),
            ey: self.ey.as_ref().expect("ey should be set after compute_fields").clone(),
            lx: self.grid.lx(),
            ly: self.grid.ly(),
        }
    }
}

/// 仿真状态快照
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub x: ndarray::Array1<f64>,
    pub y: ndarray::Array1<f64>,
    pub vx: ndarray::Array1<f64>,
    pub vy: ndarray::Array1<f64>,
    pub q: ndarray::Array1<f64>,
    pub v: Array2<f64>,
    pub ex: Array2<f64>,
    pub ey: Array2<f64>,
    /// 世界坐标范围（用于渲染归一化）
    pub lx: f64,
    pub ly: f64,
}