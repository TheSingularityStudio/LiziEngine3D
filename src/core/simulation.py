"""
主时间循环控制器，协调粒子推进和场求解的完整 PIC 模拟循环。
"""

import numpy as np
from typing import Optional, List, Dict, Any, Callable
from src.data.liziData import ParticleDataManager
from src.data.gridData import GridManager, FieldType
from src.solver.gridSolver import GridSolver
from src.solver.liziSolver import ParticleSolver


class SimulationController:
    """
    PIC 主模拟控制器。
    
    使用流程：
        1. 初始化组件 (ParticleDataManager, GridManager, GridSolver, ParticleSolver)
        2. sim = SimulationController(...)
        3. sim.run(n_steps=1000)
    
    主循环：
        for step in range(n_steps):
            particle_solver.step(dt)  # 粒子在场中运动
            grid_solver.step()        # 重新计算自生场
            diagnostics()
    """
    
    def __init__(
        self,
        particle_manager: ParticleDataManager,
        grid_manager: GridManager,
        grid_solver: GridSolver,
        particle_solver: ParticleSolver,
        dt: float = 1e-3,
        output_interval: int = 10,
        diagnostics: Optional[List[Callable]] = None
    ):
        """
        初始化模拟控制器。
        
        Args:
            particle_manager: 粒子数据管理器
            grid_manager: 网格数据管理器
            grid_solver: 网格求解器 (自生电场)
            particle_solver: 粒子求解器 (Boris 推进)
            dt: 时间步长
            output_interval: 诊断输出间隔步数
            diagnostics: 自定义诊断函数列表 [(sim, step) -> None]
        """
        self.particle_manager = particle_manager
        self.grid_manager = grid_manager
        self.grid_solver = grid_solver
        self.particle_solver = particle_solver
        
        self.dt = dt
        self.output_interval = output_interval
        self.diagnostics = diagnostics or []
        
        # 统计信息
        self.step_count = 0
        self.total_energy_history: List[float] = []
        self.total_charge_history: List[float] = []
        
        # 验证兼容性
        self._validate_components()
    
    def _validate_components(self) -> None:
        """验证组件兼容性（网格大小、场名称等）"""
        e_grid = self.grid_manager.get_grid(self.grid_solver.e_field_name)
        if e_grid is None:
            raise ValueError(f"GridManager 缺少电场 '{self.grid_solver.e_field_name}'")
        
        if self.particle_solver.e_field_name != self.grid_solver.e_field_name:
            print(f"警告: ParticleSolver E 场名 ({self.particle_solver.e_field_name}) "
                  f"与 GridSolver ({self.grid_solver.e_field_name}) 不一致")
    
    def run(self, n_steps: int) -> None:
        """
        运行 n_steps 个模拟步。
        
        Args:
            n_steps: 模拟步数
        """
        print(f"开始 PIC 模拟: {n_steps} 步, dt={self.dt}")
        print("-" * 60)
        
        for step in range(n_steps):
            # 1. 粒子推进 (在当前场中运动)
            self.particle_solver.step(self.dt)
            
            # 2. 场更新 (粒子产生新自生场)
            self.grid_solver.step(self.dt)  # dt 兼容
            
            # 3. 诊断与输出
            self.step_count = step + 1
            if step % self.output_interval == 0 or step == n_steps - 1:
                self._basic_diagnostics()
                for diag in self.diagnostics:
                    diag(self, step)
            
            # 简单进度
            if step % 100 == 0:
                print(f"步 {step}/{n_steps} 完成")
        
        print("-" * 60)
        print("模拟完成!")
    
    def _basic_diagnostics(self) -> None:
        """基本诊断：总能量、总电荷守恒"""
        # 粒子动能总和
        total_kinetic = 0.0
        for p in self.particle_manager.list_particles():
            v = np.array(p['v'])
            total_kinetic += 0.5 * p['m'] * np.dot(v, v)
        
        # 总电荷 (从 rho 网格积分)
        rho_grid = self.grid_solver.get_charge_density()
        if rho_grid:
            total_charge = 0.0
            nx, ny, nz = rho_grid.grid_size
            dx = rho_grid.cell_size
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        total_charge += rho_grid.get_vector(i, j, k)[0] * dx**3
        else:
            total_charge = sum(p['q'] for p in self.particle_manager.list_particles())
        
        self.total_energy_history.append(total_kinetic)
        self.total_charge_history.append(total_charge)
        
        print(f"步 {self.step_count:4d}: 粒子数={len(self.particle_manager.list_particles()):3d}, "
              f"动能={total_kinetic:.4e}, 总电荷={total_charge:.4e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取模拟统计"""
        return {
            'steps': self.step_count,
            'dt': self.dt,
            'n_particles': len(self.particle_manager.list_particles()),
            'total_energy': self.total_energy_history,
            'total_charge': self.total_charge_history
        }
    
    def add_diagnostic(self, diag_func: Callable) -> None:
        """添加自定义诊断函数"""
        self.diagnostics.append(diag_func)


# 示例诊断函数
def print_field_max(sim, step: int):
    """打印场最大值诊断"""
    e_grid = sim.grid_manager.get_grid(sim.grid_solver.e_field_name)
    if e_grid:
        max_e = e_grid.get_max_magnitude()
        print(f"  E_max = {max_e:.4e}")


def energy_conservation_check(sim, step: int, tol: float = 1e-10):
    """能量守恒检查"""
    energies = sim.total_energy_history
    if len(energies) > 1:
        rel_change = abs(energies[-1] - energies[0]) / energies[0]
        if rel_change > tol:
            print(f"  ⚠️  能量变化 {rel_change:.2e} (步 {step})")
        else:
            print(f"  ✓ 能量守恒良好")

