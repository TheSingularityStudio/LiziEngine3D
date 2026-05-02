"""
粒子数据管理器，管理粒子的属性（包含坐标，速度，加速度，电荷量等）。
"""


class ParticleDataManager:
    def __init__(self):
        # 初始化粒子属性
        self.particles = []  # 存储粒子信息的列表，每个粒子是一个字典，包含坐标、速度、加速度、电荷量等属性


    def add_particle(self, r, v, a, q, m):
        """
        添加一个粒子。
        :param r: 粒子的坐标 (x, y, z)
        :param v: 粒子的速度
        :param a: 粒子的加速度
        :param q: 粒子的电荷量
        :param m: 粒子的质量
        """
        particle = {
            'r': r,
            'v': v,
            'a': a,
            'q': q,
            'm': m
        }
        self.particles.append(particle)



    def update_particle(self, index, r=None, v=None, a=None, q=None, m=None):
        """
        更新指定粒子的属性。
        :param index: 粒子索引
        :param r: 新的坐标（可选）
        :param v: 新的速度（可选）
        :param a: 新的加速度（可选）
        :param q: 新的电荷量（可选）
        :param m: 新的质量（可选）
        """
        if 0 <= index < len(self.particles):
            if r is not None:
                self.particles[index]['r'] = r
            if v is not None:
                self.particles[index]['v'] = v
            if a is not None:
                self.particles[index]['a'] = a
            if q is not None:
                self.particles[index]['q'] = q
            if m is not None:
                self.particles[index]['m'] = m
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
