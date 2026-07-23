use crate::core::grid::Grid3D;
use crate::core::particles::ParticleState;

/// 半隐式欧拉时间积分（3D 版本）
pub fn step_half_implicit_euler(
    _grid: &Grid3D,
    particles: &mut ParticleState,
    dt: f64,
) {
    for p in 0..particles.len() {
        let inv_m = 1.0 / particles.m[p];
        particles.vx[p] += particles.fx[p] * inv_m * dt;
        particles.vy[p] += particles.fy[p] * inv_m * dt;
        particles.vz[p] += particles.fz[p] * inv_m * dt;
        particles.x[p] += particles.vx[p] * dt;
        particles.y[p] += particles.vy[p] * dt;
        particles.z[p] += particles.vz[p] * dt;
    }
}