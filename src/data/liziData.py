"""
粒子数据管理器，管理粒子的属性（包含加速度，速度，电荷量等）。
"""

class ParticleDataManager:
    def __init__(self):
        # 初始化粒子属性
        self.particles = []  # 存储粒子信息的列表，每个粒子是一个字典，包含加速度、速度、电荷量等属性

    def add_particle(self, acceleration, velocity, charge):
        """
        添加一个粒子。
        :param acceleration: 粒子的加速度
        :param velocity: 粒子的速度
        :param charge: 粒子的电荷量
        """
        particle = {
            'acceleration': acceleration,
            'velocity': velocity,
            'charge': charge
        }
        self.particles.append(particle)

    def update_particle(self, index, acceleration=None, velocity=None, charge=None):
        """
        更新指定粒子的属性。
        :param index: 粒子索引
        :param acceleration: 新的加速度（可选）
        :param velocity: 新的速度（可选）
        :param charge: 新的电荷量（可选）
        """
        if 0 <= index < len(self.particles):
            if acceleration is not None:
                self.particles[index]['acceleration'] = acceleration
            if velocity is not None:
                self.particles[index]['velocity'] = velocity
            if charge is not None:
                self.particles[index]['charge'] = charge
        else:
            raise IndexError("粒子索引超出范围")

    def get_particle(self, index):
        """
        获取指定粒子的属性。
        :param index: 粒子索引
        :return: 粒子的属性字典
        """
        if 0 <= index < len(self.particles):
            return self.particles[index]
        else:
            raise IndexError("粒子索引超出范围")

    def remove_particle(self, index):
        """
        移除指定的粒子。
        :param index: 粒子索引
        """
        if 0 <= index < len(self.particles):
            self.particles.pop(index)
        else:
            raise IndexError("粒子索引超出范围")

    def list_particles(self):
        """
        列出所有粒子的属性。
        :return: 粒子属性的列表
        """
        return self.particles