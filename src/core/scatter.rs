use ndarray::Array3;

use crate::core::grid::Grid3D;
use crate::core::particles::ParticleState;

/// 将粒子电荷三线性散射到网格节点上，生成离散电荷密度 rho
///
/// 每个粒子的电荷被分配到周围的 8 个网格节点上，
/// 权重由三线性插值决定。
///
/// 返回: rho, shape = (nx, ny, nz)
pub fn scatter_charges_to_grid(grid: &Grid3D, particles: &ParticleState) -> Array3<f64> {
    let (nx, ny, nz) = grid.shape();
    let mut rho = Array3::zeros((nx, ny, nz));

    for p in 0..particles.len() {
        let weight = particles.q[p];
        let (xw, yw, zw) = grid.periodic_wrap(particles.x[p], particles.y[p], particles.z[p]);
        let (gx, gy, gz) = (xw / grid.dx, yw / grid.dy, zw / grid.dz);

        let (i0, i1, j0, j1, k0, k1, wx0, wx1, wy0, wy1, wz0, wz1) =
            grid.trilinear_weights(gx, gy, gz);

        // 三线性散射到 8 个节点
        rho[[i0, j0, k0]] += weight * wx0 * wy0 * wz0;
        rho[[i1, j0, k0]] += weight * wx1 * wy0 * wz0;
        rho[[i0, j1, k0]] += weight * wx0 * wy1 * wz0;
        rho[[i1, j1, k0]] += weight * wx1 * wy1 * wz0;
        rho[[i0, j0, k1]] += weight * wx0 * wy0 * wz1;
        rho[[i1, j0, k1]] += weight * wx1 * wy0 * wz1;
        rho[[i0, j1, k1]] += weight * wx0 * wy1 * wz1;
        rho[[i1, j1, k1]] += weight * wx1 * wy1 * wz1;
    }

    rho
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid() -> Grid3D {
        Grid3D::new(4, 4, 4, 1.0, 1.0, 1.0)
    }

    #[test]
    fn test_scatter_single_charge_at_center() {
        let grid = make_grid();
        let mut p = ParticleState::zeros(1, Some(0));
        // Place at center of cell (1,1,1): world coords (1.5, 1.5, 1.5)
        p.x[0] = 1.5;
        p.y[0] = 1.5;
        p.z[0] = 1.5;
        let rho = scatter_charges_to_grid(&grid, &p);
        // Total charge should be 1.0
        let total: f64 = rho.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
        // Charge should be distributed to the 8 nodes around (1,1,1)
        // Each weight = 0.125 for a centered particle
        assert!((rho[[1, 1, 1]] - 0.125).abs() < 1e-10);
        assert!((rho[[2, 1, 1]] - 0.125).abs() < 1e-10);
        assert!((rho[[1, 2, 1]] - 0.125).abs() < 1e-10);
        assert!((rho[[2, 2, 1]] - 0.125).abs() < 1e-10);
        assert!((rho[[1, 1, 2]] - 0.125).abs() < 1e-10);
        assert!((rho[[2, 1, 2]] - 0.125).abs() < 1e-10);
        assert!((rho[[1, 2, 2]] - 0.125).abs() < 1e-10);
        assert!((rho[[2, 2, 2]] - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_scatter_total_charge_conservation() {
        let grid = make_grid();
        let n = 50;
        let mut p = ParticleState::zeros(n, Some(42));
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        for i in 0..n {
            p.x[i] = rng.gen::<f64>() * grid.lx();
            p.y[i] = rng.gen::<f64>() * grid.ly();
            p.z[i] = rng.gen::<f64>() * grid.lz();
        }
        let rho = scatter_charges_to_grid(&grid, &p);
        let total: f64 = rho.iter().sum();
        assert!((total - n as f64).abs() < 1e-10);
    }

    #[test]
    fn test_scatter_with_negative_charge() {
        let grid = make_grid();
        let mut p = ParticleState::with_charges(2, Some(0), &[1.0, -1.0]);
        p.x[0] = 0.5; p.y[0] = 0.5; p.z[0] = 0.5;
        p.x[1] = 2.5; p.y[1] = 2.5; p.z[1] = 2.5;
        let rho = scatter_charges_to_grid(&grid, &p);
        let total: f64 = rho.iter().sum();
        assert!((total - 0.0).abs() < 1e-10, "Total charge should be zero");
    }

    #[test]
    fn test_scatter_periodic_wrapping() {
        let grid = make_grid(); // lx = 4.0, ly = 4.0, lz = 4.0
        let mut p = ParticleState::zeros(1, Some(0));
        p.x[0] = 3.9;
        p.y[0] = 0.5;
        p.z[0] = 0.5;
        let rho = scatter_charges_to_grid(&grid, &p);
        let total: f64 = rho.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
        assert!(rho[[0, 0, 0]] > 0.0, "Periodic wrap: charge should appear at (0,0,0)");
    }
}