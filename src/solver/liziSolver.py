"""
粒子求解器，计算粒子运动。
使用非相对论 Boris 算法推进带电粒子在电磁场中的运动。
"""

import numpy as np
from typing import Optional, Tuple
from src.data.liziData import ParticleDataManager
from src.data.gridData import GridManager



class ParticleSolver:
    """
    粒子求解器，使用 Boris 算法计算带电粒子在电磁场中的运动。
    
    粒子受力：
        F = q * (E + v × B)
    
    使用非相对论 Boris 推进器进行时间积分，具有较好的能量守恒特性。
    """
    
    def __init__(
        self,
        particle_manager: ParticleDataManager,
        grid_manager: GridManager,
        e_field_name: str = 'E',
        b_field_name: Optional[str] = 'B',
        boundary: str = 'absorbing'
    ):
        """
        初始化粒子求解器。
        
        Args:
            particle_manager: 粒子数据管理器
            grid_manager: 网格数据管理器
            e_field_name: 电场在 GridManager 中的名称
            b_field_name: 磁场在 GridManager 中的名称，若为 None 则忽略磁场
            boundary: 边界条件类型，'absorbing'（吸收）或 'reflecting'（反射）
        """
        self.particle_manager = particle_manager
        self.grid_manager = grid_manager
        self.e_field_name = e_field_name
        self.b_field_name = b_field_name
        
        if boundary not in ('absorbing', 'reflecting'):
            raise ValueError("boundary 必须是 'absorbing' 或 'reflecting'")
        self.boundary = boundary
        
        # 获取电场网格以确定模拟域大小
        e_grid = self.grid_manager.get_grid(self.e_field_name)
        if e_grid is None:
            raise ValueError(f"电场 '{e_field_name}' 未在 GridManager 中定义")
        
        self.domain_size = (
            e_grid.grid_size[0] * e_grid.cell_size,
            e_grid.grid_size[1] * e_grid.cell_size,
            e_grid.grid_size[2] * e_grid.cell_size
        )
    
    def step(self, dt: float) -> None:
        """
        执行一个时间步长，推进所有粒子。
        
        Args:
            dt: 时间步长
        """
        e_grid = self.grid_manager.get_grid(self.e_field_name)
        b_grid = self.grid_manager.get_grid(self.b_field_name) if self.b_field_name else None
        
        particles = self.particle_manager.particles
        i = 0
        while i < len(particles):
            p = particles[i]
            
            # 提取粒子状态
            r = np.array(p['r'], dtype=float)
            v = np.array(p['v'], dtype=float)
            q = p['q']
            m = p['m']
            
            # 在粒子位置插值获取电磁场
            E = e_grid.get_field_at_position(r[0], r[1], r[2])
            B = b_grid.get_field_at_position(r[0], r[1], r[2]) if b_grid is not None else np.zeros(3)
            
            # Boris 推进
            v_new, r_new = self._boris_push(v, r, E, B, q, m, dt)
            
            # 更新加速度（用于记录）
            q_over_m = q / m
            a_new = q_over_m * (E + np.cross(v_new, B))
            
            # 更新粒子状态
            p['v'] = v_new.tolist()
            p['r'] = r_new.tolist()
            p['a'] = a_new.tolist()
            
            # 边界处理
            if self._is_outside(r_new):
                if self.boundary == 'absorbing':
                    self.particle_manager.remove_particle(i)
                    continue  # 不增加 i，下一个粒子会移到当前索引
                elif self.boundary == 'reflecting':
                    r_refl, v_refl = self._reflect(r_new, v_new)
                    p['r'] = r_refl.tolist()
                    p['v'] = v_refl.tolist()
            
            i += 1
    
    def _boris_push(
        self,
        v: np.ndarray,
        r: np.ndarray,
        E: np.ndarray,
        B: np.ndarray,
        q: float,
        m: float,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        非相对论 Boris 推进器。
        
        算法步骤：
        1. v^- = v^n + (q/m) * E * (dt/2)
        2. t = (q/m) * B * (dt/2)
        3. v' = v^- + v^- * t
        4. s = 2*t / (1 + |t|^2)
        5. v^+ = v^- + v' * s

        6. v^{n+1} = v^+ + (q/m) * E * (dt/2)
        7. r^{n+1} = r^n + (v^n + v^{n+1})/2 * dt

        
        Args:
            v: 当前速度
            r: 当前位置
            E: 电场向量
            B: 磁场向量
            q: 电荷量
            m: 质量
            dt: 时间步长
        
        Returns:
            (v_new, r_new): 更新后的速度和位置
        """
        v = np.array(v, dtype=float)
        r = np.array(r, dtype=float)
        E = np.array(E, dtype=float)
        B = np.array(B, dtype=float)
        
        q_over_m = q / m
        
        # 半步电场加速
        v_minus = v + q_over_m * E * (dt * 0.5)
        
        # 磁场旋转
        t = q_over_m * B * (dt * 0.5)
        t_mag_sq = np.dot(t, t)
        
        v_prime = v_minus + np.cross(v_minus, t)
        
        s = 2.0 * t / (1.0 + t_mag_sq)
        
        v_plus = v_minus + np.cross(v_prime, s)
        
        # 半步电场加速
        v_new = v_plus + q_over_m * E * (dt * 0.5)
        
        # 位置更新（使用平均速度以获得二阶精度）
        v_avg = (v + v_new) * 0.5
        r_new = r + v_avg * dt
        
        return v_new, r_new

    
    def _is_outside(self, r: np.ndarray) -> bool:
        """
        检查位置是否在模拟域外。
        
        Args:
            r: 位置向量
        
        Returns:
            是否在域外
        """
        return (
            r[0] < 0 or r[0] >= self.domain_size[0] or
            r[1] < 0 or r[1] >= self.domain_size[1] or
            r[2] < 0 or r[2] >= self.domain_size[2]
        )
    
    def _reflect(
        self,
        r: np.ndarray,
        v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        在边界处反射粒子和速度。
        
        Args:
            r: 当前位置
            v: 当前速度
        
        Returns:
            (r_reflected, v_reflected): 反射后的位置和速度
        """
        r = np.array(r, dtype=float)
        v = np.array(v, dtype=float)
        
        for dim in range(3):
            if r[dim] < 0:
                r[dim] = -r[dim]
                v[dim] = -v[dim]
            elif r[dim] >= self.domain_size[dim]:
                r[dim] = 2.0 * self.domain_size[dim] - r[dim]
                v[dim] = -v[dim]
        
        return r, v
