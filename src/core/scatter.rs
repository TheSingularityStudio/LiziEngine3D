use ndarray::Array2;

use crate::core::grid::Grid2D;
use crate::core::particles::ParticleState;

/// 将单位电荷粒子双线性散射到网格节点上，生成离散电荷密度 rho
///
/// 返回: rho, shape = (nx, ny)
pub fn scatter_unit_charges_to_grid(grid: &Grid2D, particles: &ParticleState) -> Array2<f64> {
    let (nx, ny) = grid.shape();
    let mut rho = Array2::zeros((nx, ny));

    for p in 0..particles.len() {
        let weight = particles.q[p]; // 使用粒子电荷量作为权重
        let (xw, yw) = grid.periodic_wrap(particles.x[p], particles.y[p]);
        let (gx, gy) = (xw / grid.dx, yw / grid.dy);

        let (i0, i1, j0, j1, wx0, wx1, wy0, wy1) = grid.bilinear_weights(gx, gy);

        rho[[i0, j0]] += weight * wx0 * wy0;
        rho[[i1, j0]] += weight * wx1 * wy0;
        rho[[i0, j1]] += weight * wx0 * wy1;
        rho[[i1, j1]] += weight * wx1 * wy1;
    }

    rho
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid() -> Grid2D {
        Grid2D::new(4, 4, 1.0, 1.0)
    }

    #[test]
    fn test_scatter_single_charge_at_center() {
        let grid = make_grid();
        let mut p = ParticleState::zeros(1, Some(0));
        // Place at center of cell (1,1): world coords (1.5, 1.5)
        p.x[0] = 1.5;
        p.y[0] = 1.5;
        let rho = scatter_unit_charges_to_grid(&grid, &p);
        // Total charge should be 1.0
        let total: f64 = rho.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
        // Charge should be distributed to the four nodes around (1,1): (1,1), (2,1), (1,2), (2,2)
        // Each weight = 0.25 for a centered particle
        assert!((rho[[1, 1]] - 0.25).abs() < 1e-10);
        assert!((rho[[2, 1]] - 0.25).abs() < 1e-10);
        assert!((rho[[1, 2]] - 0.25).abs() < 1e-10);
        assert!((rho[[2, 2]] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_scatter_total_charge_conservation() {
        let grid = make_grid();
        let n = 50;
        let mut p = ParticleState::zeros(n, Some(42));
        // Randomize positions
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        for i in 0..n {
            p.x[i] = rng.gen::<f64>() * grid.lx();
            p.y[i] = rng.gen::<f64>() * grid.ly();
        }
        let rho = scatter_unit_charges_to_grid(&grid, &p);
        let total: f64 = rho.iter().sum();
        assert!((total - n as f64).abs() < 1e-10);
    }

    #[test]
    fn test_scatter_with_negative_charge() {
        let grid = make_grid();
        let mut p = ParticleState::with_charges(2, Some(0), &[1.0, -1.0]);
        p.x[0] = 0.5; p.y[0] = 0.5;
        p.x[1] = 2.5; p.y[1] = 2.5;
        let rho = scatter_unit_charges_to_grid(&grid, &p);
        let total: f64 = rho.iter().sum();
        assert!((total - 0.0).abs() < 1e-10, "Total charge should be zero");
    }

    #[test]
    fn test_scatter_periodic_wrapping() {
        let grid = make_grid(); // lx = 4.0, ly = 4.0
        let mut p = ParticleState::zeros(1, Some(0));
        // Place at right edge near x = 4.0 (wraps to 0.0)
        p.x[0] = 3.9;
        p.y[0] = 0.5;
        let rho = scatter_unit_charges_to_grid(&grid, &p);
        let total: f64 = rho.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
        // Some charge should wrap to cell (0,0)
        assert!(rho[[0, 0]] > 0.0, "Periodic wrap: charge should appear at (0,0)");
    }
}