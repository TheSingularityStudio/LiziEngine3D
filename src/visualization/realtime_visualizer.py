import matplotlib

# 设置全局字体为支持中文的字体（如 SimHei）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

"""
实时可视化模块，使用 matplotlib 动态展示网格数据或粒子运动（3D 版本）。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class RealTimeVisualizer:
    def __init__(self, grid_manager, particle_manager):
        """
        初始化实时可视化模块。

        Args:
            grid_manager: GridManager 实例，用于获取网格数据。
            particle_manager: ParticleDataManager 实例，用于获取粒子数据。
        """
        self.grid_manager = grid_manager
        self.particle_manager = particle_manager

        # 初始化 3D 图形
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.quiver = None  # 用于显示网格向量场
        self.scatter = None  # 用于显示粒子位置

    def _update(self, frame):
        """
        更新可视化内容。

        Args:
            frame: 动画帧编号（未使用）。
        """
        self.ax.clear()

        # 可视化网格数据
        grid = self.grid_manager.get_grid('E')  # 假设电场网格名为 'E'
        if grid:
            data = grid.data
            nx, ny, nz = data.shape[:3]
            x, y, z = np.meshgrid(
                np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
            )
            u = data[:, :, :, 0]  # x 分量
            v = data[:, :, :, 1]  # y 分量
            w = data[:, :, :, 2]  # z 分量
            self.quiver = self.ax.quiver(
                x, y, z, u, v, w, length=0.5, normalize=True, color='blue'
            )

        # 可视化粒子数据
        particles = self.particle_manager.list_particles()
        if particles:
            positions = np.array([p['r'] for p in particles])  # 取 x, y, z 坐标
            self.scatter = self.ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2], c='red', s=10
            )

        self.ax.set_xlim(0, grid.grid_size[0])
        self.ax.set_ylim(0, grid.grid_size[1])
        self.ax.set_zlim(0, grid.grid_size[2])
        self.ax.set_title("实时 3D 可视化")

    def run(self, interval=100):
        """
        运行实时可视化。

        Args:
            interval: 帧间隔时间（毫秒）。
        """
        anim = FuncAnimation(self.fig, self._update, interval=interval, cache_frame_data=False)
        plt.show()