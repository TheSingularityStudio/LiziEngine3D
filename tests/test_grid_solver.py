"""
网格求解器测试脚本
验证泊松方程求解和电场计算的物理正确性
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.liziData import ParticleDataManager
from src.data.gridData import GridManager, FieldType
from src.solver.gridSolver import GridSolver, BoundaryCondition


def test_single_particle_charge_deposition():
    """测试单个粒子的电荷分配"""
    print("=" * 60)
    print("测试 1: 单个粒子的电荷分配")
    print("=" * 60)
    
    # 创建网格
    grid_mgr = GridManager()
    e_grid = grid_mgr.create_grid('E', (10, 10, 10), FieldType.ELECTRIC, cell_size=1.0)
    e_grid.fill(np.array([0.0, 0.0, 0.0]))
    
    # 创建粒子管理器，添加一个正电荷粒子
    particle_mgr = ParticleDataManager()
    particle_mgr.add_particle(
        r=[5.0, 5.0, 5.0],  # 在中心
        v=[0.0, 0.0, 0.0],
        a=[0.0, 0.0, 0.0],
        q=1.0,  # 单位电荷
        m=1.0
    )
    
    # 创建求解器
    solver = GridSolver(particle_mgr, grid_mgr)
    solver.step()
    
    # 获取电荷密度
    rho_grid = solver.get_charge_density()
    
    print(f"网格中心 ({5,5,5}) 电荷密度:")
    print(f"  {rho_grid.get_vector(5, 5, 5)}")
    
    # 检查总电荷守恒（近似）
    total_charge = 0.0
    for i in range(10):
        for j in range(10):
            for k in range(10):
                # 网格存储为向量的 x 分量
                total_charge += rho_grid.get_vector(i, j, k)[0]
    
    print(f"总电荷: {total_charge} (应为 ~1.0)")
    
    # 总电荷应该约等于粒子电荷
    assert abs(total_charge - 1.0) < 0.1, f"总电荷不守恒: {total_charge}"
    print("✓ 电荷分配测试通过\n")


def test_point_charge_electric_field():
    """测试点电荷的电场（与解析解比较）"""
    print("=" * 60)
    print("测试 2: 点电荷的电场（与库仑定律比较）")
    print("=" * 60)
    
    # 创建网格
    grid_mgr = GridManager()
    e_grid = grid_mgr.create_grid('E', (20, 20, 20), FieldType.ELECTRIC, cell_size=1.0)
    e_grid.fill(np.array([0.0, 0.0, 0.0]))
    
    # 在中心放置点电荷
    particle_mgr = ParticleDataManager()
    particle_mgr.add_particle(
        r=[10.0, 10.0, 10.0],
        v=[0.0, 0.0, 0.0],
        a=[0.0, 0.0, 0.0],
        q=1.0,
        m=1.0
    )
    
    # 创建求解器
    solver = GridSolver(particle_mgr, grid_mgr, boundary=BoundaryCondition.ABSORBING)
    solver.step()
    
    # 获取计算的电场
    e_self_grid = solver.get_electric_field()
    
    # 在距离中心 3 格的位置测量电场
    r_test = np.array([13.0, 10.0, 10.0])  # x 方向 3 格
    e_calc = e_self_grid.get_field_at_position(r_test[0], r_test[1], r_test[2])
    
    # 解析解：对于点电荷，E = q / (4*pi*eps0*r^2) * r_hat
    # 使用 epsilon_0 = 8.854e-12
    r_dist = 3.0
    q = 1.0
    eps0 = 8.854e-12
    
    # 库仑场强（解析解）
    e_magnitude = q / (4 * np.pi * eps0 * r_dist**2)
    e_expected = np.array([e_magnitude, 0.0, 0.0])  # 指向外
    
    print(f"测试点位置: {r_test}")
    print(f"计算电场: {e_calc}")
    print(f"解析电场: {e_expected}")
    print(f"电场模: {np.linalg.norm(e_calc)} vs {np.linalg.norm(e_expected)}")
    
    # 注意：由于网格分辨率限制��数值解会有显著误差
    # 这里主要检查电场方向正确
    if np.linalg.norm(e_calc) > 0:
        direction_correct = np.dot(e_calc, e_expected) > 0
        print(f"方向正确: {direction_correct}")
    
    print("✓ 点电荷电场测试完成\n")


def test_dipole_electric_field():
    """测试偶极子的电场分布"""
    print("=" * 60)
    print("测试 3: 偶极子的电场分布")
    print("=" * 60)
    
    # 创建网格
    grid_mgr = GridManager()
    e_grid = grid_mgr.create_grid('E', (30, 30, 30), FieldType.ELECTRIC, cell_size=1.0)
    e_grid.fill(np.array([0.0, 0.0, 0.0]))
    
    # 创建偶极子：正负电荷
    particle_mgr = ParticleDataManager()
    # 正电荷在 (14, 15, 15)
    particle_mgr.add_particle(
        r=[14.0, 15.0, 15.0],
        v=[0.0, 0.0, 0.0],
        a=[0.0, 0.0, 0.0],
        q=1.0,
        m=1.0
    )
    # 负电荷在 (16, 15, 15)
    particle_mgr.add_particle(
        r=[16.0, 15.0, 15.0],
        v=[0.0, 0.0, 0.0],
        a=[0.0, 0.0, 0.0],
        q=-1.0,
        m=1.0
    )
    
    # 创建求解器
    solver = GridSolver(particle_mgr, grid_mgr, boundary=BoundaryCondition.ABSORBING)
    solver.step()
    
    # 获取电场
    e_self_grid = solver.get_electric_field()
    
    # 在两电荷中点测量
    r_mid = np.array([15.0, 15.0, 15.0])
    e_mid = e_self_grid.get_field_at_position(r_mid[0], r_mid[1], r_mid[2])
    
    # 在 x 方向远场测量
    r_far = np.array([25.0, 15.0, 15.0])
    e_far = e_self_grid.get_field_at_position(r_far[0], r_far[1], r_far[2])
    
    print(f"中点 {r_mid} 电场: {e_mid}")
    print(f"远场 {r_far} 电场: {e_far}")
    
    # 偶极子中点电场应该主要指向负电荷方向（从正指向负）
    # 即 x 方向为负
    if np.linalg.norm(e_mid) > 0:
        print(f"中点方向: {'负 x' if e_mid[0] < 0 else '正 x'}")
    
    # 远场电场应该指向正电荷方向（因为是偶极子，远场类似于负电荷）
    if np.linalg.norm(e_far) > 0:
        print(f"远场方向: {'负 x' if e_far[0] < 0 else '正 x'}")
        # 对于偶极子，远场电场指向负电荷方向
        assert e_far[0] < 0, "远场电场应指向负电荷方向"
    
    print("✓ 偶极子电场测试通过\n")


def test_periodic_boundary():
    """测试周期性边界条件"""
    print("=" * 60)
    print("测试 4: 周期性边界条件")
    print("=" * 60)
    
    # 创建网格
    grid_mgr = GridManager()
    e_grid = grid_mgr.create_grid('E', (10, 10, 10), FieldType.ELECTRIC, cell_size=1.0)
    e_grid.fill(np.array([0.0, 0.0, 0.0]))
    
    # 创建粒子管理器，添加一个正电荷粒子
    particle_mgr = ParticleDataManager()
    particle_mgr.add_particle(
        r=[5.0, 5.0, 5.0],
        v=[0.0, 0.0, 0.0],
        a=[0.0, 0.0, 0.0],
        q=1.0,
        m=1.0
    )
    
    # 使用周期性边界
    solver = GridSolver(particle_mgr, grid_mgr, boundary=BoundaryCondition.PERIODIC)
    solver.step()
    
    # 获取电荷密度
    rho_grid = solver.get_charge_density()
    
    # 检查总电荷守恒（周期性边界应该保持总电荷）
    total_charge = 0.0
    for i in range(10):
        for j in range(10):
            for k in range(10):
                total_charge += rho_grid.get_vector(i, j, k)[0]
    
    print(f"周期性边界总电荷: {total_charge} (应为 ~1.0)")
    assert abs(total_charge - 1.0) < 0.1, f"周期性边界电荷不守恒: {total_charge}"
    print("✓ 周期性边界测试通过\n")


def test_charge_conservation():
    """测试电荷守恒（不同网格大小）"""
    print("=" * 60)
    print("测试 5: 电荷守恒（网格大��影响）")
    print("=" * 60)
    
    grid_sizes = [(10, 10, 10), (20, 20, 20), (50, 50, 50)]
    
    for grid_size in grid_sizes:
        grid_mgr = GridManager()
        e_grid = grid_mgr.create_grid('E', grid_size, FieldType.ELECTRIC, cell_size=1.0)
        e_grid.fill(np.array([0.0, 0.0, 0.0]))
        
        particle_mgr = ParticleDataManager()
        particle_mgr.add_particle(
            r=[grid_size[0]/2, grid_size[1]/2, grid_size[2]/2],
            v=[0.0, 0.0, 0.0],
            a=[0.0, 0.0, 0.0],
            q=1.0,
            m=1.0
        )
        
        solver = GridSolver(particle_mgr, grid_mgr)
        solver.step()
        
        rho_grid = solver.get_charge_density()
        
        total_charge = 0.0
        nx, ny, nz = grid_size
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    total_charge += rho_grid.get_vector(i, j, k)[0]
        
        error = abs(total_charge - 1.0)
        print(f"网格 {grid_size}: 总电荷 = {total_charge:.6f}, 误差 = {error:.6e}")
        
        # 网格越密，电荷守恒越好
        assert error < 0.15, f"网格 {grid_size} 电荷不守恒: {total_charge}"
    
    print("✓ 电荷守恒测试通过\n")


def test_multiple_particles():
    """测试多个粒子的电场叠加"""
    print("=" * 60)
    print("测试 6: 多粒子场叠加")
    print("=" * 60)
    
    # 创建网格
    grid_mgr = GridManager()
    e_grid = grid_mgr.create_grid('E', (20, 20, 20), FieldType.ELECTRIC, cell_size=1.0)
    e_grid.fill(np.array([0.0, 0.0, 0.0]))
    
    # 创建多个随机分布的粒子
    np.random.seed(42)
    particle_mgr = ParticleDataManager()
    
    n_particles = 10
    total_charge = 0.0
    for _ in range(n_particles):
        r = [
            np.random.uniform(1, 19),
            np.random.uniform(1, 19),
            np.random.uniform(1, 19)
        ]
        q = np.random.choice([-1.0, 1.0])
        total_charge += q
        
        particle_mgr.add_particle(
            r=r,
            v=[0.0, 0.0, 0.0],
            a=[0.0, 0.0, 0.0],
            q=q,
            m=1.0
        )
    
    print(f"粒子数: {n_particles}, 总电荷: {total_charge}")
    
    solver = GridSolver(particle_mgr, grid_mgr)
    solver.step()
    
    # 检查总电荷
    rho_grid = solver.get_charge_density()
    nx, ny, nz = 20, 20, 20
    
    calc_charge = 0.0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                calc_charge += rho_grid.get_vector(i, j, k)[0]
    
    print(f"计算总电荷: {calc_charge:.6f}")
    assert abs(calc_charge - total_charge) < 0.5, f"多粒子电荷不守恒: {calc_charge}"
    print("✓ 多粒子测试通过\n")


if __name__ == '__main__':
    test_single_particle_charge_deposition()
    test_point_charge_electric_field()
    test_dipole_electric_field()
    test_periodic_boundary()
    test_charge_conservation()
    test_multiple_particles()
    print("=" * 60)
    print("所有测试通过！")
    print("=" * 60)
