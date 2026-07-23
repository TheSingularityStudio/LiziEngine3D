use ndarray::Array1;
use ndarray::Array3;

use crate::core::grid::Grid3D;
use crate::core::particles::ParticleState;

/// 将定义在网格节点上的 3D 电场矢量场 (Ex, Ey, Ez) 通过三线性插值采样到粒子位置
///
/// 返回: (fx, fy, fz) — shape 均为 (N,) 的粒子受力分量
pub fn gather_field_to_particles_trilinear(
    grid: &Grid3D,
    particles: &ParticleState,
    ex: &Array3<f64>,
    ey: &Array3<f64>,
    ez: &Array3<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n = particles.len();
    let mut fx = Array1::zeros(n);
    let mut fy = Array1::zeros(n);
    let mut fz = Array1::zeros(n);

    for p in 0..n {
        let (xw, yw, zw) = grid.periodic_wrap(particles.x[p], particles.y[p], particles.z[p]);
        let (gx, gy, gz) = (xw / grid.dx, yw / grid.dy, zw / grid.dz);
        let (i0, i1, j0, j1, k0, k1, wx0, wx1, wy0, wy1, wz0, wz1) =
            grid.trilinear_weights(gx, gy, gz);

        fx[p] = ex[[i0, j0, k0]] * wx0 * wy0 * wz0
            + ex[[i1, j0, k0]] * wx1 * wy0 * wz0
            + ex[[i0, j1, k0]] * wx0 * wy1 * wz0
            + ex[[i1, j1, k0]] * wx1 * wy1 * wz0
            + ex[[i0, j0, k1]] * wx0 * wy0 * wz1
            + ex[[i1, j0, k1]] * wx1 * wy0 * wz1
            + ex[[i0, j1, k1]] * wx0 * wy1 * wz1
            + ex[[i1, j1, k1]] * wx1 * wy1 * wz1;

        fy[p] = ey[[i0, j0, k0]] * wx0 * wy0 * wz0
            + ey[[i1, j0, k0]] * wx1 * wy0 * wz0
            + ey[[i0, j1, k0]] * wx0 * wy1 * wz0
            + ey[[i1, j1, k0]] * wx1 * wy1 * wz0
            + ey[[i0, j0, k1]] * wx0 * wy0 * wz1
            + ey[[i1, j0, k1]] * wx1 * wy0 * wz1
            + ey[[i0, j1, k1]] * wx0 * wy1 * wz1
            + ey[[i1, j1, k1]] * wx1 * wy1 * wz1;

        fz[p] = ez[[i0, j0, k0]] * wx0 * wy0 * wz0
            + ez[[i1, j0, k0]] * wx1 * wy0 * wz0
            + ez[[i0, j1, k0]] * wx0 * wy1 * wz0
            + ez[[i1, j1, k0]] * wx1 * wy1 * wz0
            + ez[[i0, j0, k1]] * wx0 * wy0 * wz1
            + ez[[i1, j0, k1]] * wx1 * wy0 * wz1
            + ez[[i0, j1, k1]] * wx0 * wy1 * wz1
            + ez[[i1, j1, k1]] * wx1 * wy1 * wz1;
    }
    (fx, fy, fz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_uniform_field() {
        let grid = Grid3D::new(4, 4, 4, 1.0, 1.0, 1.0);
        let (nx, ny, nz) = grid.shape();
        let ex = Array3::from_elem((nx, ny, nz), 2.0);
        let ey = Array3::from_elem((nx, ny, nz), 3.0);
        let ez = Array3::from_elem((nx, ny, nz), 4.0);
        let mut particles = ParticleState::zeros(3, Some(0));
        particles.x[0] = 0.5; particles.y[0] = 0.5; particles.z[0] = 0.5;
        particles.x[1] = 1.5; particles.y[1] = 2.5; particles.z[1] = 1.5;
        particles.x[2] = 3.0; particles.y[2] = 3.0; particles.z[2] = 3.0;
        let (fx, fy, fz) = gather_field_to_particles_trilinear(&grid, &particles, &ex, &ey, &ez);
        for i in 0..3 {
            assert!((fx[i] - 2.0).abs() < 1e-10);
            assert!((fy[i] - 3.0).abs() < 1e-10);
            assert!((fz[i] - 4.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gather_at_grid_node() {
        let grid = Grid3D::new(4, 4, 4, 1.0, 1.0, 1.0);
        let mut ex = Array3::zeros((4, 4, 4));
        let mut ey = Array3::zeros((4, 4, 4));
        let mut ez = Array3::zeros((4, 4, 4));
        ex[[0, 0, 0]] = 5.0;
        ey[[1, 1, 1]] = 7.0;
        ez[[2, 2, 2]] = 9.0;
        let mut particles = ParticleState::zeros(3, Some(0));
        particles.x[0] = 0.0; particles.y[0] = 0.0; particles.z[0] = 0.0;
        particles.x[1] = 1.0; particles.y[1] = 1.0; particles.z[1] = 1.0;
        particles.x[2] = 2.0; particles.y[2] = 2.0; particles.z[2] = 2.0;
        let (fx, fy, fz) = gather_field_to_particles_trilinear(&grid, &particles, &ex, &ey, &ez);
        assert!((fx[0] - 5.0).abs() < 1e-10);
        assert!((fy[0] - 0.0).abs() < 1e-10);
        assert!((fz[0] - 0.0).abs() < 1e-10);
        assert!((fx[1] - 0.0).abs() < 1e-10);
        assert!((fy[1] - 7.0).abs() < 1e-10);
        assert!((fz[1] - 0.0).abs() < 1e-10);
        assert!((fx[2] - 0.0).abs() < 1e-10);
        assert!((fy[2] - 0.0).abs() < 1e-10);
        assert!((fz[2] - 9.0).abs() < 1e-10);
    }
}