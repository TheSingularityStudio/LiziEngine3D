use ndarray::Array1;
use ndarray::Array2;

use crate::core::grid::Grid2D;
use crate::core::particles::ParticleState;

/// 将定义在网格节点上的电场矢量场 (Ex, Ey) 通过双线性插值采样到粒子位置
///
/// 返回: (fx, fy) — shape 均为 (N,) 的粒子受力分量
pub fn gather_field_to_particles_bilinear(
    grid: &Grid2D,
    particles: &ParticleState,
    ex: &Array2<f64>,
    ey: &Array2<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let n = particles.len();

    let mut fx = Array1::zeros(n);
    let mut fy = Array1::zeros(n);

    for p in 0..n {
        let (xw, yw) = grid.periodic_wrap(particles.x[p], particles.y[p]);
        let (gx, gy) = (xw / grid.dx, yw / grid.dy);

        let (i0, i1, j0, j1, wx0, wx1, wy0, wy1) = grid.bilinear_weights(gx, gy);

        // 双线性插值
        fx[p] = ex[[i0, j0]] * wx0 * wy0
            + ex[[i1, j0]] * wx1 * wy0
            + ex[[i0, j1]] * wx0 * wy1
            + ex[[i1, j1]] * wx1 * wy1;

        fy[p] = ey[[i0, j0]] * wx0 * wy0
            + ey[[i1, j0]] * wx1 * wy0
            + ey[[i0, j1]] * wx0 * wy1
            + ey[[i1, j1]] * wx1 * wy1;
    }

    (fx, fy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_uniform_field() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let (nx, ny) = grid.shape();
        let ex = Array2::from_elem((nx, ny), 2.0);
        let ey = Array2::from_elem((nx, ny), 3.0);

        let mut particles = ParticleState::zeros(3, Some(0));
        particles.x[0] = 0.5; particles.y[0] = 0.5;
        particles.x[1] = 1.5; particles.y[1] = 2.5;
        particles.x[2] = 3.0; particles.y[2] = 3.0;

        let (fx, fy) = gather_field_to_particles_bilinear(&grid, &particles, &ex, &ey);
        for i in 0..3 {
            assert!((fx[i] - 2.0).abs() < 1e-10, "fx[{}] should be 2.0, got {}", i, fx[i]);
            assert!((fy[i] - 3.0).abs() < 1e-10, "fy[{}] should be 3.0, got {}", i, fy[i]);
        }
    }

    #[test]
    fn test_gather_at_grid_node() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0);
        let mut ex = Array2::zeros((4, 4));
        let mut ey = Array2::zeros((4, 4));
        ex[[0, 0]] = 5.0; // Only node (0,0) has value
        ey[[1, 1]] = 7.0; // Only node (1,1) has value

        let mut particles = ParticleState::zeros(2, Some(0));
        // Particle exactly at node (0,0): world = (0.0, 0.0)
        particles.x[0] = 0.0;
        particles.y[0] = 0.0;
        // Particle exactly at node (1,1): world = (1.0, 1.0)
        particles.x[1] = 1.0;
        particles.y[1] = 1.0;

        let (fx, fy) = gather_field_to_particles_bilinear(&grid, &particles, &ex, &ey);
        assert!((fx[0] - 5.0).abs() < 1e-10, "Particle at (0,0) should get ex=5.0");
        assert!((fy[0] - 0.0).abs() < 1e-10, "Particle at (0,0) should get ey=0.0");
        assert!((fx[1] - 0.0).abs() < 1e-10, "Particle at (1,1) should get ex=0.0");
        assert!((fy[1] - 7.0).abs() < 1e-10, "Particle at (1,1) should get ey=7.0");
    }

    #[test]
    fn test_gather_periodic_wrapping() {
        let grid = Grid2D::new(4, 4, 1.0, 1.0); // lx = 4.0, ly = 4.0
        let mut ex = Array2::zeros((4, 4));
        ex[[0, 2]] = 10.0; // Only node (0,2) has non-zero
        let ey = Array2::zeros((4, 4)); // E_y is all zeros

        let mut particles = ParticleState::zeros(1, Some(0));
        // Place particle near x=4.0 (wraps to x=0.0), y=2.0
        particles.x[0] = 3.9;
        particles.y[0] = 2.0;

        let (fx, _) = gather_field_to_particles_bilinear(&grid, &particles, &ex, &ey);
        // Particle wraps to near (0,2), should get ~10.0
        assert!((fx[0] - 10.0).abs() > 1e-10, "Periodic wrap should affect interpolation");
    }
}