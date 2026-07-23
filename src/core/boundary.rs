use crate::core::grid::Grid2D;
use crate::core::particles::ParticleState;

/// 边界类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundaryType {
    #[default]
    Periodic,      // 周期边界
    Reflective,    // 反弹边界（完全弹性）
    Open,          // 虚空边界（移出即删除）
}

impl BoundaryType {
    pub fn all() -> [BoundaryType; 3] {
        [
            BoundaryType::Periodic,
            BoundaryType::Reflective,
            BoundaryType::Open,
        ]
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            BoundaryType::Periodic => "周期边界",
            BoundaryType::Reflective => "反弹边界",
            BoundaryType::Open => "虚空边界",
        }
    }
}

/// 应用边界条件到粒子
///
/// 根据边界类型，处理粒子的位置和速度：
/// - Periodic: 周期包裹（从一边出去从另一边进来）
/// - Reflective: 反弹边界（完全弹性碰撞，速度反转）
pub fn apply_boundary_conditions(
    particles: &mut ParticleState,
    grid: &Grid2D,
    boundary_type: BoundaryType,
) {
    let lx = grid.lx();
    let ly = grid.ly();

    match boundary_type {
        BoundaryType::Periodic => {
            // 周期包裹
            for p in 0..particles.len() {
                particles.x[p] = ((particles.x[p] % lx) + lx) % lx;
                particles.y[p] = ((particles.y[p] % ly) + ly) % ly;
            }
        }
        BoundaryType::Reflective => {
            // 反弹边界（完全弹性碰撞）
            for p in 0..particles.len() {
                // X 方向
                if particles.x[p] < 0.0 {
                    particles.x[p] = -particles.x[p];
                    particles.vx[p] = -particles.vx[p];
                } else if particles.x[p] >= lx {
                    particles.x[p] = 2.0 * lx - particles.x[p];
                    particles.vx[p] = -particles.vx[p];
                }

                // Y 方向
                if particles.y[p] < 0.0 {
                    particles.y[p] = -particles.y[p];
                    particles.vy[p] = -particles.vy[p];
                } else if particles.y[p] >= ly {
                    particles.y[p] = 2.0 * ly - particles.y[p];
                    particles.vy[p] = -particles.vy[p];
                }

                // 确保粒子在边界内（处理数值误差）
                particles.x[p] = particles.x[p].clamp(0.0, lx);
                particles.y[p] = particles.y[p].clamp(0.0, ly);
            }
        }
        BoundaryType::Open => {
            // 虚空边界：删除超出 [0, lx) × [0, ly) 的粒子
            let len = particles.len();
            let mut to_remove: Vec<usize> = Vec::new();
            for p in 0..len {
                if particles.x[p] < 0.0 || particles.x[p] >= lx
                    || particles.y[p] < 0.0 || particles.y[p] >= ly
                {
                    to_remove.push(p);
                }
            }
            // 从后往前删除，避免索引偏移
            for &idx in to_remove.iter().rev() {
                particles.remove_particle(idx);
            }
        }
    }
}

/// 应用最高速度限制
///
/// 如果粒子的速度大小超过 max_speed，则将其速度向量缩放至 max_speed。
/// 这可以防止粒子速度无限增长，保持模拟稳定性。
pub fn apply_speed_limit(
    particles: &mut ParticleState,
    max_speed: f64,
) {
    if max_speed <= 0.0 {
        return;
    }

    let max_speed_sq = max_speed * max_speed;

    for p in 0..particles.len() {
        let speed_sq = particles.vx[p] * particles.vx[p] + particles.vy[p] * particles.vy[p];
        
        if speed_sq > max_speed_sq {
            let speed = speed_sq.sqrt();
            let factor = max_speed / speed;
            particles.vx[p] *= factor;
            particles.vy[p] *= factor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_grid() -> Grid2D {
        Grid2D::new(10, 10, 1.0, 1.0)
    }

    fn create_test_particles() -> ParticleState {
        ParticleState {
            x: ndarray::arr1(&[5.0, 5.0, 5.0]),
            y: ndarray::arr1(&[5.0, 5.0, 5.0]),
            vx: ndarray::arr1(&[1.0, 0.0, 0.0]),
            vy: ndarray::arr1(&[0.0, 1.0, 0.0]),
            fx: ndarray::arr1(&[0.0, 0.0, 0.0]),
            fy: ndarray::arr1(&[0.0, 0.0, 0.0]),
            q: ndarray::arr1(&[1.0, 1.0, 1.0]),
            m: ndarray::arr1(&[1.0, 1.0, 1.0]),
        }
    }

    #[test]
    fn test_periodic_boundary() {
        let grid = create_test_grid();
        let mut particles = create_test_particles();
        
        // 移动粒子超出边界
        particles.x[0] = 12.0;  // 超出右边界
        particles.y[1] = -3.0;  // 超出下边界

        apply_boundary_conditions(&mut particles, &grid, BoundaryType::Periodic);

        assert!((particles.x[0] - 2.0).abs() < 1e-10);
        assert!((particles.y[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_reflective_boundary() {
        let grid = create_test_grid();
        let mut particles = create_test_particles();
        
        // 移动粒子超出右边界
        particles.x[0] = 12.0;
        particles.vx[0] = 2.0;

        apply_boundary_conditions(&mut particles, &grid, BoundaryType::Reflective);

        // 位置应该反弹回来：2*10 - 12 = 8
        assert!((particles.x[0] - 8.0).abs() < 1e-10);
        // 速度应该反转
        assert!((particles.vx[0] + 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_speed_limit() {
        let mut particles = create_test_particles();
        
        // 设置一个很大的速度
        particles.vx[0] = 6.0;
        particles.vy[0] = 8.0;  // 速度大小 = 10

        apply_speed_limit(&mut particles, 5.0);

        let speed = (particles.vx[0].powi(2) + particles.vy[0].powi(2)).sqrt();
        assert!((speed - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_speed_limit_no_change_when_under_limit() {
        let mut particles = create_test_particles();
        
        particles.vx[0] = 1.0;
        particles.vy[0] = 2.0;  // 速度大小 ≈ 2.24

        let original_vx = particles.vx[0];
        let original_vy = particles.vy[0];

        apply_speed_limit(&mut particles, 5.0);

        assert!((particles.vx[0] - original_vx).abs() < 1e-10);
        assert!((particles.vy[0] - original_vy).abs() < 1e-10);
    }
}