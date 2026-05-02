"""
粒子求解器测试脚本
验证 Boris 算法在匀强电场、匀强磁场及边界条件下的物理正确性
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.liziData import ParticleDataManager
from src.data.gridData import GridManager, FieldType
from src.solver.liziSolver import ParticleSolver


def test_uniform_e_field():
    """测试匀强电场中的匀加速运动"""
    print("=" * 60)
    print("测试 1: 匀强电场中的匀加速运动")
    print("=" * 60)
    
    # 创建 100x10x10 网格，电场 E = (1, 0, 0)
    grid_mgr = GridManager()
    e_grid = grid_mgr.create_grid('E', (100, 10, 10), FieldType.ELECTRIC, cell_size=1.0)
    e_grid.fill(np.array([1.0, 0.0, 0.0]))
    
    # 创建粒子管理器，添加一个静止粒子
    particle_mgr = ParticleDataManager()
    particle_mgr.add_particle(
        r=[5.0, 5.0, 5.0],
        v=[0.0, 0.0, 0.0],
        a=[0.0, 0.0, 0.0],
        q=1.0,
        m=1.0
    )
    
    # 创建求解器（无磁场）
    solver = ParticleSolver(particle_mgr, grid_mgr, b_field_name=None)
    
    # 模拟参数
    dt = 0.01
    n_steps = 1000
    total_time = dt * n_steps  # 10.0
    
    # 推进
    for _ in range(n_steps):
        solver.step(dt)
    
    p = particle_mgr.get_particle(0)
    r = np.array(p['r'])
    v = np.array(p['v'])
    
    # 理论值: v = qE/m * t = 1 * 10 = 10
    #         r = r0 + 0.5 * qE/m * t^2 = 5 + 0.5 * 100 = 55
    v_expected = np.array([10.0, 0.0, 0.0])
    r_expected = np.array([55.0, 5.0, 5.0])
    
    v_error = np.linalg.norm(v - v_expected)
    r_error = np.linalg.norm(r - r_expected)
    
    print(f"模拟后速度: {v}")
    print(f"理论速度:   {v_expected}")
    print(f"速度误差:   {v_error:.6e}")
    print(f"模拟后位置: {r}")
    print(f"理论位置:   {r_expected}")
    print(f"位置误差:   {r_error:.6e}")
    
    assert v_error < 1e-10, f"速度误差过大: {v_error}"
    assert r_error < 1e-10, f"位置误差过大: {r_error}"
    print("✓ 匀强电场测试通过\n")


def test_uniform_b_field():
    """测试匀强磁场中的回旋运动与能量守恒"""
    print("=" * 60)
    print("测试 2: 匀强磁场中的回旋运动与能量守恒")
    print("=" * 60)
    
    # 创建 20x20x20 网格，磁场 B = (0, 0, 1)
    grid_mgr = GridManager()
    e_grid = grid_mgr.create_grid('E', (20, 20, 20), FieldType.ELECTRIC, cell_size=1.0)
    e_grid.fill(np.array([0.0, 0.0, 0.0]))
    
    b_grid = grid_mgr.create_grid('B', (20, 20, 20), FieldType.MAGNETIC, cell_size=1.0)
    b_grid.fill(np.array([0.0, 0.0, 1.0]))
    
    # 创建粒子管理器，添加一个具有垂直于 B 方向速度的粒子
    particle_mgr = ParticleDataManager()
    particle_mgr.add_particle(
        r=[10.0, 10.0, 10.0],
        v=[1.0, 0.0, 0.0],
        a=[0.0, 0.0, 0.0],
        q=1.0,
        m=1.0
    )
    
    # 创建求解器
    solver = ParticleSolver(particle_mgr, grid_mgr)
    
    # 模拟参数
    dt = 0.01
    n_steps = 1000  # 10 个时间单位，约 1.59 个周期
    
    # 记录能量
    energies = []
    
    # 推进
    for _ in range(n_steps):
        solver.step(dt)
        p = particle_mgr.get_particle(0)
        v = np.array(p['v'])
        energies.append(np.dot(v, v))
    
    p = particle_mgr.get_particle(0)
    r_final = np.array(p['r'])
    v_final = np.array(p['v'])
    
    # 检查能量守恒
    energy_variation = np.max(energies) - np.min(energies)
    energy_relative = energy_variation / np.mean(energies)
    
    print(f"初始能量:   {energies[0]:.6f}")
    print(f"最终能量:   {energies[-1]:.6f}")
    print(f"能量最大偏差: {energy_variation:.6e}")
    print(f"能量相对偏差: {energy_relative:.6e}")
    print(f"最终位置:   {r_final}")
    print(f"最终速度:   {v_final}")
    
    # Boris 算法应极好地保持能量守恒
    assert energy_relative < 1e-12, f"能量不守恒: 相对偏差 {energy_relative}"
    print("✓ 匀强磁场测试通过\n")


def test_absorbing_boundary():
    """测试吸收边界条件"""
    print("=" * 60)
    print("测试 3: 吸收边界条件")
    print("=" * 60)
    
    # 创建 10x10x10 网格
    grid_mgr = GridManager()
    e_grid = grid_mgr.create_grid('E', (10, 10, 10), FieldType.ELECTRIC, cell_size=1.0)
    e_grid.fill(np.array([0.0, 0.0, 0.0]))
    
    particle_mgr = ParticleDataManager()
    # 粒子靠近左边界，向左运动
    particle_mgr.add_particle(
        r=[0.5, 5.0, 5.0],
        v=[-1.0, 0.0, 0.0],
        a=[0.0, 0.0, 0.0],
        q=1.0,
        m=1.0
    )
    
    solver = ParticleSolver(particle_mgr, grid_mgr, b_field_name=None, boundary='absorbing')
    
    dt = 0.6
    # 一步就会出界
    solver.step(dt)
    
    n_particles = len(particle_mgr.list_particles())
    print(f"边界步进后粒子数: {n_particles}")
    
    assert n_particles == 0, f"吸收边界应移除粒子，但剩余 {n_particles} 个"
    print("✓ 吸收边界测试通过\n")


def test_reflecting_boundary():
    """测试反射边界条件"""
    print("=" * 60)
    print("测试 4: 反射边界条件")
    print("=" * 60)
    
    # 创建 10x10x10 网格
    grid_mgr = GridManager()
    e_grid = grid_mgr.create_grid('E', (10, 10, 10), FieldType.ELECTRIC, cell_size=1.0)
    e_grid.fill(np.array([0.0, 0.0, 0.0]))
    
    particle_mgr = ParticleDataManager()
    # 粒子非常靠近左边界，向左运动，一步会出界
    particle_mgr.add_particle(
        r=[0.05, 5.0, 5.0],
        v=[-1.0, 0.0, 0.0],
        a=[0.0, 0.0, 0.0],
        q=1.0,
        m=1.0
    )
    
    solver = ParticleSolver(particle_mgr, grid_mgr, b_field_name=None, boundary='reflecting')
    
    dt = 0.1
    solver.step(dt)
    
    p = particle_mgr.get_particle(0)
    r = np.array(p['r'])
    v = np.array(p['v'])
    
    print(f"反射后位置: {r}")
    print(f"反射后速度: {v}")
    
    # 位置应在域内
    assert r[0] >= 0, f"反射后位置应在域内: r[0]={r[0]}"
    # 速度 x 分量应反向
    assert v[0] > 0, f"反射后速度应反向: v[0]={v[0]}"
    print("✓ 反射边界测试通过\n")


if __name__ == '__main__':
    test_uniform_e_field()
    test_uniform_b_field()
    test_absorbing_boundary()
    test_reflecting_boundary()
    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)
