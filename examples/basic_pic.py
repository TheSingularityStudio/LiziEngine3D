"""
基本 PIC 模拟示例：2 个反向电荷粒子的静电相互作用。
验证主时间循环器工作。
"""

import sys
import os
import numpy as np

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.liziData import ParticleDataManager
from src.data.gridData import GridManager, FieldType
from src.solver.gridSolver import GridSolver, BoundaryCondition
from src.solver.liziSolver import ParticleSolver
from src.core.simulation import SimulationController, print_field_max, energy_conservation_check
from src.visualization.realtime_visualizer import RealTimeVisualizer
import threading


def main():
    """运行基本 PIC 模拟"""
    print("=== LiziEngine3D 基本 PIC 示例 ===")
    
    # 1. 创建网格 (小网格快速测试)
    grid_mgr = GridManager()
    grid_mgr.create_grid('E', (16, 16, 16), FieldType.ELECTRIC, cell_size=1.0)
    grid_mgr.create_grid('rho', (16, 16, 16), FieldType.ELECTRIC_POTENTIAL, cell_size=1.0)
    
    # 2. 创建粒子：正负电荷对，初始分离
    particle_mgr = ParticleDataManager()
    particle_mgr.add_particle(  # 正电荷 (左)
        r=[4.0, 8.0, 8.0], v=[0.1, 0.0, 0.0], a=[0,0,0], q=+1.0, m=1.0
    )
    particle_mgr.add_particle(  # 负电荷 (右)
        r=[12.0, 8.0, 8.0], v=[-0.1, 0.0, 0.0], a=[0,0,0], q=-1.0, m=1.0
    )
    
    # 3. 创建求解器
    grid_solver = GridSolver(
        particle_manager=particle_mgr,
        grid_manager=grid_mgr,
        e_field_name='E',
        rho_name='rho',
        boundary=BoundaryCondition.PERIODIC  # 周期边界避免逃逸
    )
    
    particle_solver = ParticleSolver(
        particle_manager=particle_mgr,
        grid_manager=grid_mgr,
        e_field_name='E',
        boundary='reflecting'
    )
    
    # 4. 创建模拟控制器
    sim = SimulationController(
        particle_manager=particle_mgr,
        grid_manager=grid_mgr,
        grid_solver=grid_solver,
        particle_solver=particle_solver,
        dt=0.01,  # 小步长确保稳定
        output_interval=2
    )
    
    # 添加诊断
    sim.add_diagnostic(print_field_max)
    sim.add_diagnostic(energy_conservation_check)

    # 创建实时可视化器
    visualizer = RealTimeVisualizer(grid_mgr, particle_mgr)

    # 在后台线程中运行模拟
    def run_simulation():
        sim.run(n_steps=20)

    sim_thread = threading.Thread(target=run_simulation)
    sim_thread.start()

    # 在主线程中运行可视化
    visualizer.run(interval=100)

    print("\n✅ 示例完成！检查输出：粒子吸引 + 能量守恒。")


if __name__ == "__main__":
    main()
