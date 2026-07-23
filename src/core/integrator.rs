use crate::core::grid::Grid2D;
use crate::core::particles::ParticleState;

/// 半隐式欧拉时间积分：
///
///   v_{n+1} = v_n + a_n * dt
///   x_{n+1} = x_n + v_{n+1} * dt
///
/// 加速度 a = F / m
///
/// 注意：边界处理由调用方（`apply_boundary_conditions`）根据边界类型执行，
/// 此处不做任何边界包裹，允许粒子自由移动。
pub fn step_half_implicit_euler(
    _grid: &Grid2D,
    particles: &mut ParticleState,
    dt: f64,
) {
    for p in 0..particles.len() {
        // v += a * dt (a = F / m)
        let inv_m = 1.0 / particles.m[p];
        particles.vx[p] += particles.fx[p] * inv_m * dt;
        particles.vy[p] += particles.fy[p] * inv_m * dt;

        // x += v * dt
        particles.x[p] += particles.vx[p] * dt;
        particles.y[p] += particles.vy[p] * dt;
        // 边界处理由调用方的 apply_boundary_conditions 完成
    }
}