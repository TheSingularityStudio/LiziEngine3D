"""
网格求解器，计算粒子产生的自生电场。
使用泊松方程求解电荷分布产生的电势，然后计算电场。
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple
from enum import Enum

from src.data.liziData import ParticleDataManager
from src.data.gridData import GridManager, GridData, FieldType


class BoundaryCondition(Enum):
    """边界条件类型"""
    ABSORBING = "absorbing"      # 吸收边界：电场在边界处逐渐消失
    PERIODIC = "periodic"        # 周期性边界：电场从一边出去从另一边回来
    OPEN = "open"                # 开放边界：模拟向无限空间延伸


class GridSolver:
    """
    网格求解器，使用泊松方程计算粒子产生的自生电场。
    
    流程：
    1. 电荷分配（CIC）：将粒子电荷分配到网格节点
    2. 泊松方程求解：∇²φ = -ρ/ε₀
    3. 电场计算：E = -∇φ
    
    属性:
        epsilon_0: 真空介电常数（约 8.854e-12 F/m）
    """
    
    def __init__(
        self,
        particle_manager: ParticleDataManager,
        grid_manager: GridManager,
        e_field_name: str = 'E_self',
        rho_name: str = 'rho',
        boundary: BoundaryCondition = BoundaryCondition.ABSORBING,
        epsilon_0: float = 8.854e-12
    ):
        """
        初始化网格求解器。
        
        Args:
            particle_manager: 粒子数据管理器
            grid_manager: 网格数据管理器
            e_field_name: 自生电场在 GridManager 中的名称
            rho_name: 电荷密度网格的名称
            boundary: 边界条件类型
            epsilon_0: 真空介电常数
        """
        self.particle_manager = particle_manager
        self.grid_manager = grid_manager
        self.e_field_name = e_field_name
        self.rho_name = rho_name
        self.boundary = boundary
        self.epsilon_0 = epsilon_0
        
        # 获取参考网格以确定模拟域大小
        ref_grid = None
        for name in grid_manager.grid_names:
            grid = grid_manager.get_grid(name)
            if grid is not None:
                ref_grid = grid
                break
        
        if ref_grid is None:
            raise ValueError("GridManager 中没有可用的网格作为参考")
        
        self.grid_size = ref_grid.grid_size
        self.cell_size = ref_grid.cell_size
        
        # 预计算拉普拉斯算子矩阵
        self._laplacian_matrix = None
        
        # 创建电荷密度网格
        self._create_rho_grid()
        
        # 创建电场网格
        self._create_e_field_grid()
    
    def _create_rho_grid(self) -> None:
        """创建电荷密度网格"""
        if not self.grid_manager.has_grid(self.rho_name):
            self.grid_manager.create_grid(
                self.rho_name,
                self.grid_size,
                FieldType.ELECTRIC_POTENTIAL,
                self.cell_size
            )
    
    def _create_e_field_grid(self) -> None:
        """创建电场网格"""
        if not self.grid_manager.has_grid(self.e_field_name):
            self.grid_manager.create_grid(
                self.e_field_name,
                self.grid_size,
                FieldType.ELECTRIC,
                self.cell_size
            )
    
    def step(self, dt: Optional[float] = None) -> None:
        """
        执行一个时间步长，计算粒子的自生电场。
        
        Args:
            dt: 时间步长（可选，用于与 ParticleSolver 兼容）
        """
        # Step 1: 电荷分配
        self._deposit_charge()
        
        # Step 2: 求解泊松方程
        phi = self._solve_poisson()
        
        # Step 3: 计算电场
        self._compute_electric_field(phi)
    
    def _deposit_charge(self) -> None:
        """
        使用 CIC (Cloud-in-Cell) 方法将粒子电荷分配到网格节点。
        
        每个粒子的电荷按照权重分配到周围最近的 8 个网格点。
        """
        rho_grid = self.grid_manager.get_grid(self.rho_name)
        rho_grid.clear()
        
        nx, ny, nz = self.grid_size
        dx, dy, dz = self.cell_size, self.cell_size, self.cell_size
        
        for particle in self.particle_manager.particles:
            r = np.array(particle['r'], dtype=float)
            q = particle['q']
            
            # 将物理坐标转换为网格索引
            fi = r[0] / dx
            fj = r[1] / dy
            fk = r[2] / dz
            
            i = int(fi)
            j = int(fj)
            k = int(fk)
            
            # 计算分数部分
            dx_local = fi - i
            dy_local = fj - j
            dz_local = fk - k
            
            # CIC 权重（8个相邻网格点）
            weights = np.array([
                (1 - dx_local) * (1 - dy_local) * (1 - dz_local),
                (1 - dx_local) * (1 - dy_local) * dz_local,
                (1 - dx_local) * dy_local * (1 - dz_local),
                (1 - dx_local) * dy_local * dz_local,
                dx_local * (1 - dy_local) * (1 - dz_local),
                dx_local * (1 - dy_local) * dz_local,
                dx_local * dy_local * (1 - dz_local),
                dx_local * dy_local * dz_local,
            ])
            
            # 8个相邻网格点的索引
            indices = [
                (i, j, k), (i, j, k+1),
                (i, j+1, k), (i, j+1, k+1),
                (i+1, j, k), (i+1, j, k+1),
                (i+1, j+1, k), (i+1, j+1, k+1)
            ]
            
            # 分配电荷到网格点
            for (idx, w) in zip(indices, weights):
                ii, jj, kk = idx
                if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                    charge_density = q * w / (dx * dy * dz)
                    current = rho_grid.get_vector(ii, jj, kk)
                    rho_grid.set_vector(ii, jj, kk, current + np.array([charge_density, 0, 0]))
    
    def _build_laplacian_matrix(self) -> sparse.csr_matrix:
        """
        构建离散拉普拉斯算子的稀疏矩阵。
        
        使用七点 stencil:
        φ(i+1,j,k) + φ(i-1,j,k) + φ(i,j+1,k) + φ(i,j-1,k) + φ(i,j,k+1) + φ(i,j,k-1) - 6*φ(i,j,k)
        -------------------------------dx²
        = rho(i,j,k) / epsilon_0
        """
        if self._laplacian_matrix is not None:
            return self._laplacian_matrix
        
        nx, ny, nz = self.grid_size
        n = nx * ny * nz
        
        def idx(i: int, j: int, k: int) -> int:
            return i * (ny * nz) + j * nz + k
        
        def is_boundary(i: int, j: int, k: int) -> bool:
            return (i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1)
        
        di = []
        dj = []
        dv = []
        
        dx2 = self.cell_size ** 2
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    n_idx = idx(i, j, k)
                    
                    # 吸收边界：边界点使用单位矩阵
                    if self.boundary == BoundaryCondition.ABSORBING and is_boundary(i, j, k):
                        di.append(n_idx)
                        dj.append(n_idx)
                        dv.append(1.0)
                        continue
                    
                    # 内部点
                    coeff_center = -6.0 / dx2
                    
                    di.append(n_idx)
                    dj.append(n_idx)
                    dv.append(coeff_center)
                    
                    # x 方向相邻点
                    if i > 0:
                        di.append(n_idx)
                        dj.append(idx(i-1, j, k))
                        dv.append(1.0 / dx2)
                    elif self.boundary == BoundaryCondition.PERIODIC:
                        di.append(n_idx)
                        dj.append(idx(nx-1, j, k))
                        dv.append(1.0 / dx2)
                    
                    if i < nx - 1:
                        di.append(n_idx)
                        dj.append(idx(i+1, j, k))
                        dv.append(1.0 / dx2)
                    elif self.boundary == BoundaryCondition.PERIODIC:
                        di.append(n_idx)
                        dj.append(idx(0, j, k))
                        dv.append(1.0 / dx2)
                    
                    # y 方向相邻点
                    if j > 0:
                        di.append(n_idx)
                        dj.append(idx(i, j-1, k))
                        dv.append(1.0 / dx2)
                    elif self.boundary == BoundaryCondition.PERIODIC:
                        di.append(n_idx)
                        dj.append(idx(i, ny-1, k))
                        dv.append(1.0 / dx2)
                    
                    if j < ny - 1:
                        di.append(n_idx)
                        dj.append(idx(i, j+1, k))
                        dv.append(1.0 / dx2)
                    elif self.boundary == BoundaryCondition.PERIODIC:
                        di.append(n_idx)
                        dj.append(idx(i, 0, k))
                        dv.append(1.0 / dx2)
                    
                    # z 方向相邻点
                    if k > 0:
                        di.append(n_idx)
                        dj.append(idx(i, j, k-1))
                        dv.append(1.0 / dx2)
                    elif self.boundary == BoundaryCondition.PERIODIC:
                        di.append(n_idx)
                        dj.append(idx(i, j, nz-1))
                        dv.append(1.0 / dx2)
                    
                    if k < nz - 1:
                        di.append(n_idx)
                        dj.append(idx(i, j, k+1))
                        dv.append(1.0 / dx2)
                    elif self.boundary == BoundaryCondition.PERIODIC:
                        di.append(n_idx)
                        dj.append(idx(i, j, 0))
                        dv.append(1.0 / dx2)
        
        self._laplacian_matrix = sparse.csr_matrix(
            (dv, (di, dj)), shape=(n, n)
        )
        
        return self._laplacian_matrix
    
    def _solve_poisson(self) -> np.ndarray:
        """
        求解泊松方程：∇²φ = -ρ/ε₀
        
        Returns:
            电势数组，形状为 (nx, ny, nz)
        """
        nx, ny, nz = self.grid_size
        
        # 获取电荷密度
        rho_grid = self.grid_manager.get_grid(self.rho_name)
        rho = np.zeros((nx, ny, nz))
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    rho[i, j, k] = rho_grid.get_vector(i, j, k)[0]
        
        # 构建右侧向量
        rhs = -rho.flatten() / self.epsilon_0
        
        # 吸收边界：边界点右侧设为 0
        if self.boundary == BoundaryCondition.ABSORBING:
            def idx(i: int, j: int, k: int) -> int:
                return i * (ny * nz) + j * nz + k
            
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
                            n_idx = idx(i, j, k)
                            rhs[n_idx] = 0.0
        
        # 构建拉普拉斯矩阵并求解
        laplacian = self._build_laplacian_matrix()
        phi_flat = spsolve(laplacian, rhs)
        
        # 重塑为 3D 数组
        phi = phi_flat.reshape((nx, ny, nz))
        
        return phi
    
    def _compute_electric_field(self, phi: np.ndarray) -> None:
        """
        从电势计算电场：E = -∇φ
        
        Args:
            phi: 电势数组，形状为 (nx, ny, nz)
        """
        nx, ny, nz = self.grid_size
        dx = self.cell_size
        
        e_grid = self.grid_manager.get_grid(self.e_field_name)
        e_grid.clear()
        
        # 使用中心差分计算电场
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # x 方向
                    if i == 0:
                        ex = (phi[1, j, k] - phi[0, j, k]) / dx
                    elif i == nx - 1:
                        ex = (phi[nx-1, j, k] - phi[nx-2, j, k]) / dx
                    else:
                        ex = (phi[i+1, j, k] - phi[i-1, j, k]) / (2 * dx)
                    
                    # y 方向
                    if j == 0:
                        ey = (phi[i, 1, k] - phi[i, 0, k]) / dx
                    elif j == ny - 1:
                        ey = (phi[i, ny-1, k] - phi[i, ny-2, k]) / dx
                    else:
                        ey = (phi[i, j+1, k] - phi[i, j-1, k]) / (2 * dx)
                    
                    # z 方向
                    if k == 0:
                        ez = (phi[i, j, 1] - phi[i, j, 0]) / dx
                    elif k == nz - 1:
                        ez = (phi[i, j, nz-1] - phi[i, j, nz-2]) / dx
                    else:
                        ez = (phi[i, j, k+1] - phi[i, j, k-1]) / (2 * dx)
                    
                    # 电场 = -∇φ
                    e_field = np.array([-ex, -ey, -ez])
                    e_grid.set_vector(i, j, k, e_field)
    
    def get_electric_field(self) -> Optional[GridData]:
        """获取计算得到的自生电场网格"""
        return self.grid_manager.get_grid(self.e_field_name)
    
    def get_charge_density(self) -> Optional[GridData]:
        """获取电荷密度网格"""
        return self.grid_manager.get_grid(self.rho_name)
    
    def __repr__(self) -> str:
        return (
            f"GridSolver(grid_size={self.grid_size}, "
            f"boundary={self.boundary.value}, "
            f"epsilon_0={self.epsilon_0})"
        )
