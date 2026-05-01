"""
网格数据管理器，管理作为"电场""磁场"等的网格。
"""

import numpy as np
from enum import Enum
from typing import Optional, Tuple


class FieldType(Enum):
    """场的类型枚举"""
    ELECTRIC = "electric"      # 电场
    MAGNETIC = "magnetic"      # 磁场
    ELECTRIC_POTENTIAL = "electric_potential"  # 电势
    MAGNETIC_POTENTIAL = "magnetic_potential"  # 磁势


class GridData:
    """网格数据管理器，管理三维网格中的向量场数据"""
    
    def __init__(
        self,
        grid_size: Tuple[int, int, int],
        field_type: FieldType = FieldType.ELECTRIC,
        cell_size: float = 1.0
    ):
        """
        初始化网格数据管理器
        
        Args:
            grid_size: 网格大小 (nx, ny, nz)
            field_type: 场的类型
            cell_size: 网格单元的物理大小
        """
        self._grid_size = grid_size
        self._field_type = field_type
        self._cell_size = cell_size
        
        # 创建三维向量数组，存储每个网格点的向量值 (Ex, Ey, Ez)
        self._data = np.zeros((grid_size[0], grid_size[1], grid_size[2], 3))
    
    @property
    def grid_size(self) -> Tuple[int, int, int]:
        """获取网格大小"""
        return self._grid_size
    
    @property
    def field_type(self) -> FieldType:
        """获取场类型"""
        return self._field_type
    
    @property
    def cell_size(self) -> float:
        """获取网格单元大小"""
        return self._cell_size
    
    @property
    def data(self) -> np.ndarray:
        """获取网格数据数组"""
        return self._data
    
    def get_vector(self, i: int, j: int, k: int) -> np.ndarray:
        """
        获取指定位置的向量值
        
        Args:
            i, j, k: 网格索引
        
        Returns:
            三维向量数组 [Ex, Ey, Ez]
        """
        if not self._is_valid_index(i, j, k):
            raise IndexError(f"索引 ({i}, {j}, {k}) 超出网格范围 {self._grid_size}")
        return self._data[i, j, k].copy()
    
    def set_vector(self, i: int, j: int, k: int, vector: np.ndarray) -> None:
        """
        设置指定位置的向量值
        
        Args:
            i, j, k: 网格索引
            vector: 三维向量值 [Ex, Ey, Ez]
        """
        if not self._is_valid_index(i, j, k):
            raise IndexError(f"索引 ({i}, {j}, {k}) 超出网格范围 {self._grid_size}")
        if len(vector) != 3:
            raise ValueError("向量必须是三维的")
        self._data[i, j, k] = vector
    
    def get_field_at_position(
        self, 
        x: float, 
        y: float, 
        z: float,
        zero_outside: bool = True
    ) -> np.ndarray:
        """
        根据物理坐标获取场向量值（使用三线性插值）
        
        Args:
            x, y, z: 物理坐标
            zero_outside: 是否在网格外部返回零向量
        
        Returns:
            插值后的三维向量
        """
        # 将物理坐标转换为网格索引
        fi = x / self._cell_size
        fj = y / self._cell_size
        fk = z / self._cell_size
        
        i0, j0, k0 = int(fi), int(fj), int(fk)
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        
        # 获取分数部分
        dx, dy, dz = fi - i0, fj - j0, fk - k0
        
        # 如果在网格外部
        if zero_outside and (
            i0 < 0 or i1 >= self._grid_size[0] or
            j0 < 0 or j1 >= self._grid_size[1] or
            k0 < 0 or k1 >= self._grid_size[2]
        ):
            return np.zeros(3)
        
        # 确保索引在有效范围内
        i0 = max(0, min(i0, self._grid_size[0] - 1))
        j0 = max(0, min(j0, self._grid_size[1] - 1))
        k0 = max(0, min(k0, self._grid_size[2] - 1))
        i1 = max(0, min(i1, self._grid_size[0] - 1))
        j1 = max(0, min(j1, self._grid_size[1] - 1))
        k1 = max(0, min(k1, self._grid_size[2] - 1))
        
        # 三线性插值
        result = np.zeros(3)
        for dim in range(3):
            c000 = self._data[i0, j0, k0][dim]
            c001 = self._data[i0, j0, k1][dim]
            c010 = self._data[i0, j1, k0][dim]
            c011 = self._data[i0, j1, k1][dim]
            c100 = self._data[i1, j0, k0][dim]
            c101 = self._data[i1, j0, k1][dim]
            c110 = self._data[i1, j1, k0][dim]
            c111 = self._data[i1, j1, k1][dim]
            
            c00 = c000 * (1 - dx) + c100 * dx
            c01 = c001 * (1 - dx) + c101 * dx
            c10 = c010 * (1 - dx) + c110 * dx
            c11 = c011 * (1 - dx) + c111 * dx
            
            c0 = c00 * (1 - dy) + c10 * dy
            c1 = c01 * (1 - dy) + c11 * dy
            
            result[dim] = c0 * (1 - dz) + c1 * dz
        
        return result
    
    def clear(self) -> None:
        """清除所有网格数据"""
        self._data.fill(0.0)
    
    def fill(self, vector: np.ndarray) -> None:
        """
        用指定向量填充整个网格
        
        Args:
            vector: 要填充的三维向量
        """
        if len(vector) != 3:
            raise ValueError("向量必须是三维的")
        self._data[:] = vector
    
    def add_vector(self, i: int, j: int, k: int, vector: np.ndarray) -> None:
        """
        向指定位置添加向量值
        
        Args:
            i, j, k: 网格索引
            vector: 要添加的三维向量
        """
        if not self._is_valid_index(i, j, k):
            raise IndexError(f"索引 ({i}, {j}, {k}) 超出网格范围 {self._grid_size}")
        if len(vector) != 3:
            raise ValueError("向量必须是三维的")
        self._data[i, j, k] += vector
    
    def get_magnitude(self, i: int, j: int, k: int) -> float:
        """
        获取指定位置向量的模
        
        Args:
            i, j, k: 网格索引
        
        Returns:
            向量的模
        """
        if not self._is_valid_index(i, j, k):
            raise IndexError(f"索引 ({i}, {j}, {k}) 超出网格范围 {self._grid_size}")
        return np.linalg.norm(self._data[i, j, k])
    
    def get_total_magnitude(self) -> float:
        """
        获取整个网格中所有向量的总模（所有向量模的和）
        
        Returns:
            总模
        """
        return np.sum(np.linalg.norm(self._data, axis=3))
    
    def get_max_magnitude(self) -> float:
        """
        获取整个网格中向量的最大模
        
        Returns:
            最大模
        """
        return np.max(np.linalg.norm(self._data, axis=3))
    
    def get_max_magnitude_position(self) -> Tuple[int, int, int]:
        """
        获取最大模向量的位置索引
        
        Returns:
            位置索引 (i, j, k)
        """
        magnitudes = np.linalg.norm(self._data, axis=3)
        max_idx = np.argmax(magnitudes)
        return np.unravel_index(max_idx, magnitudes.shape)
    
    def _is_valid_index(self, i: int, j: int, k: int) -> bool:
        """
        检查索引是否在有效范围内
        
        Args:
            i, j, k: 网格索引
        
        Returns:
            是否有效
        """
        return (
            0 <= i < self._grid_size[0] and
            0 <= j < self._grid_size[1] and
            0 <= k < self._grid_size[2]
        )
    
    def get_slice(self, axis: str, index: int) -> np.ndarray:
        """
        获取指定轴的切片数据
        
        Args:
            axis: 轴名 ('x', 'y', 'z')
            index: 切片索引
        
        Returns:
            切片数据，形状为 (ny, nz, 3) 或类似形式
        """
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        ax = axis_map.get(axis.lower())
        if ax is None:
            raise ValueError(f"无效的轴名: {axis}，必须是 'x', 'y', 或 'z'")
        if not (0 <= index < self._grid_size[ax]):
            raise IndexError(f"索引 {index} 超出轴 {axis} 的范围 {self._grid_size[ax]}")
        
        if ax == 0:
            return self._data[index, :, :, :].copy()
        elif ax == 1:
            return self._data[:, index, :, :].copy()
        else:
            return self._data[:, :, index, :].copy()
    
    def apply_function(self, func) -> None:
        """
        对每个网格点应用函数
        
        Args:
            func: 接收位置 (i, j, k) 和当前向量，返回新向量的函数
        """
        for i in range(self._grid_size[0]):
            for j in range(self._grid_size[1]):
                for k in range(self._grid_size[2]):
                    self._data[i, j, k] = func(i, j, k, self._data[i, j, k])
    
    def copy(self) -> 'GridData':
        """
        创建网格数据的深拷贝
        
        Returns:
            新的GridData实例
        """
        new_grid = GridData(self._grid_size, self._field_type, self._cell_size)
        new_grid._data = self._data.copy()
        return new_grid
    
    def __repr__(self) -> str:
        return (
            f"GridData(grid_size={self._grid_size}, "
            f"field_type={self._field_type.value}, "
            f"cell_size={self._cell_size})"
        )


class GridManager:
    """网格数据管理器，管理多个网格场"""
    
    def __init__(self):
        """初始化网格管理器"""
        self._grids: dict[str, GridData] = {}
    
    def add_grid(self, name: str, grid: GridData) -> None:
        """
        添加网格到管理器
        
        Args:
            name: 网格名称
            grid: GridData实例
        """
        self._grids[name] = grid
    
    def remove_grid(self, name: str) -> None:
        """
        移除网格
        
        Args:
            name: 网格名称
        """
        if name in self._grids:
            del self._grids[name]
    
    def get_grid(self, name: str) -> Optional[GridData]:
        """
        获取指定名称的网格
        
        Args:
            name: 网格名称
        
        Returns:
            GridData实例，如果不存在返回None
        """
        return self._grids.get(name)
    
    def has_grid(self, name: str) -> bool:
        """
        检查是否存在指定名称的网格
        
        Args:
            name: 网格名称
        
        Returns:
            是否存在
        """
        return name in self._grids
    
    @property
    def grid_names(self) -> list[str]:
        """获取所有网格名称"""
        return list(self._grids.keys())
    
    @property
    def grid_count(self) -> int:
        """获取网格数量"""
        return len(self._grids)
    
    def create_grid(
        self,
        name: str,
        grid_size: Tuple[int, int, int],
        field_type: FieldType = FieldType.ELECTRIC,
        cell_size: float = 1.0
    ) -> GridData:
        """
        创建并添加新网格
        
        Args:
            name: 网格名称
            grid_size: 网格大小
            field_type: 场类型
            cell_size: 单元大小
        
        Returns:
            创建的GridData实例
        """
        grid = GridData(grid_size, field_type, cell_size)
        self._grids[name] = grid
        return grid
    
    def clear_all(self) -> None:
        """清除所有网格数据"""
        for grid in self._grids.values():
            grid.clear()
    
    def __getitem__(self, name: str) -> GridData:
        """通过名称获取网格"""
        return self._grids[name]
    
    def __contains__(self, name: str) -> bool:
        """检查网格是否存在"""
        return name in self._grids
    
    def __len__(self) -> int:
        """获取网格数量"""
        return len(self._grids)
    
    def __repr__(self) -> str:
        return f"GridManager(grids={list(self._grids.keys())})"
