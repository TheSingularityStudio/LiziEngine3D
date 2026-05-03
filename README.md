# LiziEngine3D 🚀
三维粒子模拟引擎 (PIC 方法)，支持电磁场 + Boris 粒子推进。

## ✨ 新功能：主时间循环器已添加！
- `src/core/simulation.py`: SimulationController - 完整 PIC 循环。
- 基本诊断：能量/电荷守恒、场统计。

## 🚀 快速启动 (1 分钟)
```bash
cd c:/Users/22739/Documents/Dev/LiziEngine3D
python examples/basic_pic.py
```
**预期输出**：
```
开始 PIC 模拟: 20 步, dt=0.01
步    0: 粒子数=  2, 动能=1.0100e-03, 总电荷=0.0000e+00
步    2: ... (粒子吸引，E 场增加)
...
模拟完成!
能量变化: ~1e-12 (守恒良好)
```

## 📋 核心组件
```
src/
├── core/simulation.py     # 主控制器 (新增)
├── data/                  # 粒子/网格数据
│   ├── liziData.py
│   └── gridData.py
└── solver/                # 求解器
    ├── gridSolver.py      # 静电场 (Poisson)
    └── liziSolver.py      # 粒子 Boris 推进
```

## 🎯 扩展计划 (TODO.md)
- 磁场求解器
- HDF5 输出 + 可视化
- GPU 并行

## 🔧 依赖
- numpy
- scipy

**试试 basic_pic.py，查看粒子相互吸引！**
