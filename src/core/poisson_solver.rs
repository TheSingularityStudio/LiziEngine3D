use std::fmt;
use ndarray::{Array3, Axis};
use ndrustfft::{ndfft, ndifft, FftHandler};
use num_complex::Complex64;
use num_traits::Zero;

/// 使用 FFT 的 3D Poisson 求解器（带缓存，避免每帧重新分配 FFT Handler）
///
/// 在周期域中求解离散 Poisson 方程: ∇²V = -rho
/// 在傅里叶空间: V_hat(k) = rho_hat(k) / (kx² + ky² + kz²)
pub struct PoissonSolver {
    nx: usize,
    ny: usize,
    nz: usize,
    handler_0: FftHandler<f64>,
    handler_1: FftHandler<f64>,
    handler_2: FftHandler<f64>,
    k2: Array3<f64>,
}

impl PoissonSolver {
    /// 创建新的求解器，预先分配 FFT Handler 和 k² 矩阵
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64, _eps: f64) -> Self {
        let handler_0 = FftHandler::new(nx);
        let handler_1 = FftHandler::new(ny);
        let handler_2 = FftHandler::new(nz);

        let mut k2 = Array3::<f64>::zeros((nx, ny, nz));
        for i in 0..nx {
            let freq_i = if i <= nx / 2 {
                i as f64
            } else {
                i as f64 - nx as f64
            };
            let kx = 2.0 * std::f64::consts::PI * freq_i / (nx as f64 * dx);
            for j in 0..ny {
                let freq_j = if j <= ny / 2 {
                    j as f64
                } else {
                    j as f64 - ny as f64
                };
                let ky = 2.0 * std::f64::consts::PI * freq_j / (ny as f64 * dy);
                for k in 0..nz {
                    let freq_k = if k <= nz / 2 {
                        k as f64
                    } else {
                        k as f64 - nz as f64
                    };
                    let kz = 2.0 * std::f64::consts::PI * freq_k / (nz as f64 * dz);
                    k2[[i, j, k]] = kx * kx + ky * ky + kz * kz;
                }
            }
        }

        Self { nx, ny, nz, handler_0, handler_1, handler_2, k2 }
    }

    /// 求解 Poisson 方程
    ///
    /// 参数:
    /// - rho: 电荷密度, shape (nx, ny, nz)
    /// - eps: k² ≈ 0 模式的阈值，避免除零
    ///
    /// 返回: 电势 V, shape (nx, ny, nz)
    pub fn solve(&self, rho: &Array3<f64>, eps: f64) -> Array3<f64> {
        // 转换为复数
        let mut rho_hat: Array3<Complex64> = rho.mapv(|v| Complex64::new(v, 0.0));

        // 3D FFT: 沿 axis=0, 再 axis=1, 再 axis=2
        let mut tmp = Array3::zeros((self.nx, self.ny, self.nz));

        ndfft(&rho_hat, &mut tmp, &self.handler_0, 0);
        ndfft(&tmp, &mut rho_hat, &self.handler_1, 1);
        ndfft(&rho_hat, &mut tmp, &self.handler_2, 2);

        // V_hat = rho_hat / k² (k≠0 时), k≈0 时设为 0
        for i in 0..self.nx {
            for j in 0..self.ny {
                for k in 0..self.nz {
                    if self.k2[[i, j, k]] > eps {
                        rho_hat[[i, j, k]] = rho_hat[[i, j, k]] / self.k2[[i, j, k]];
                    } else {
                        rho_hat[[i, j, k]] = Complex64::zero();
                    }
                }
            }
        }

        // 逆 3D FFT: 沿 axis=2, 再 axis=1, 再 axis=0
        let mut tmp2 = Array3::zeros((self.nx, self.ny, self.nz));
        let mut v_complex: Array3<Complex64> = Array3::zeros((self.nx, self.ny, self.nz));

        ndifft(&rho_hat, &mut tmp2, &self.handler_2, 2);
        ndifft(&tmp2, &mut v_complex, &self.handler_1, 1);
        ndifft(&v_complex, &mut tmp2, &self.handler_0, 0);

        // 取实部
        tmp2.mapv(|c| c.re)
    }

    /// 获取网格尺寸
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
}

impl fmt::Debug for PoissonSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoissonSolver")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .field("k2", &self.k2)
            .finish()
    }
}

/// 在周期边界下使用中心差分计算 3D 电场
///
/// E = -∇V
/// E_x(i,j,k) = -(V(i+1,j,k) - V(i-1,j,k)) / (2dx)
/// E_y(i,j,k) = -(V(i,j+1,k) - V(i,j-1,k)) / (2dy)
/// E_z(i,j,k) = -(V(i,j,k+1) - V(i,j,k-1)) / (2dz)
///
/// 返回: (Ex, Ey, Ez)，均为 shape (nx, ny, nz)
pub fn compute_e_from_potential_periodic(
    v: &Array3<f64>,
    dx: f64,
    dy: f64,
    dz: f64,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    // E_x: -(V(i+1,j,k) - V(i-1,j,k)) / (2dx)
    let vx_p = shift_array3(v, Axis(0), 1isize);
    let vx_m = shift_array3(v, Axis(0), -1isize);
    let ex = -(vx_p - vx_m) / (2.0 * dx);

    // E_y: -(V(i,j+1,k) - V(i,j-1,k)) / (2dy)
    let vy_p = shift_array3(v, Axis(1), 1isize);
    let vy_m = shift_array3(v, Axis(1), -1isize);
    let ey = -(vy_p - vy_m) / (2.0 * dy);

    // E_z: -(V(i,j,k+1) - V(i,j,k-1)) / (2dz)
    let vz_p = shift_array3(v, Axis(2), 1isize);
    let vz_m = shift_array3(v, Axis(2), -1isize);
    let ez = -(vz_p - vz_m) / (2.0 * dz);

    (ex, ey, ez)
}

/// 在指定轴上滚动 3D 数组（类似 np.roll）
pub fn shift_array3(arr: &Array3<f64>, axis: Axis, shift: isize) -> Array3<f64> {
    let dim = arr.dim();
    let mut result = Array3::zeros(dim);

    let len = match axis {
        Axis(0) => dim.0,
        Axis(1) => dim.1,
        Axis(2) => dim.2,
        _ => unreachable!(),
    };
    let len_i = len as isize;

    match axis {
        Axis(0) => {
            for idx in 0..len {
                let src_idx = ((idx as isize + shift).rem_euclid(len_i)) as usize;
                for j in 0..dim.1 {
                    for k in 0..dim.2 {
                        result[[idx, j, k]] = arr[[src_idx, j, k]];
                    }
                }
            }
        }
        Axis(1) => {
            for idx in 0..len {
                let src_idx = ((idx as isize + shift).rem_euclid(len_i)) as usize;
                for i in 0..dim.0 {
                    for k in 0..dim.2 {
                        result[[i, idx, k]] = arr[[i, src_idx, k]];
                    }
                }
            }
        }
        Axis(2) => {
            for idx in 0..len {
                let src_idx = ((idx as isize + shift).rem_euclid(len_i)) as usize;
                for i in 0..dim.0 {
                    for j in 0..dim.1 {
                        result[[i, j, idx]] = arr[[i, j, src_idx]];
                    }
                }
            }
        }
        _ => unreachable!(),
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_solver_uniform_rho_zero_laplacian() {
        let solver = PoissonSolver::new(8, 8, 8, 1.0, 1.0, 1.0, 1e-12);
        let rho = Array3::from_elem((8, 8, 8), 1.0);
        let v = solver.solve(&rho, 1e-12);
        // In 3D FFT with periodic BC, uniform rho should produce near-zero V
        // (small residual due to finite grid and FFT normalization)
        let max_abs = v.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert!(max_abs < 1.0, "Uniform rho should give near-zero V, max={}", max_abs);
    }

    #[test]
    fn test_poisson_solver_point_charge() {
        let nx = 16;
        let ny = 16;
        let nz = 16;
        let dx = 1.0;
        let dy = 1.0;
        let dz = 1.0;
        let solver = PoissonSolver::new(nx, ny, nz, dx, dy, dz, 1e-12);
        let mut rho = Array3::zeros((nx, ny, nz));
        rho[[8, 8, 8]] = 1.0 / (dx * dy * dz);

        let v = solver.solve(&rho, 1e-12);

        assert!(v[[8, 8, 8]] > 0.0, "V at charge should be positive");
        assert!(v[[8, 8, 8]] > v[[4, 4, 4]], "V should decay with distance");
    }

    #[test]
    fn test_electric_field_from_poisson_dipole() {
        let nx = 16;
        let ny = 16;
        let nz = 16;
        let dx = 1.0;
        let dy = 1.0;
        let dz = 1.0;
        let solver = PoissonSolver::new(nx, ny, nz, dx, dy, dz, 1e-12);
        let mut rho = Array3::zeros((nx, ny, nz));
        rho[[6, 8, 8]] = 1.0 / (dx * dy * dz);
        rho[[10, 8, 8]] = -1.0 / (dx * dy * dz);

        let v = solver.solve(&rho, 1e-12);
        let (ex, ey, ez) = compute_e_from_potential_periodic(&v, dx, dy, dz);

        assert!(ex[[8, 8, 8]] > 0.0, "E_x should point from + to - charge");
        assert!(ey[[8, 8, 8]].abs() < 1.0, "E_y should be small along symmetry axis");
        assert!(ez[[8, 8, 8]].abs() < 1.0, "E_z should be small along symmetry axis");
    }
}