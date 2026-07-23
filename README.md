# LiziEngine2D 🧪⚡

**2D 静电 PIC (Particle-In-Cell) 模拟器** — 基于 Rust 实现，使用 CPU 并行计算电场与粒子运动的实时交互仿真。

[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

---

## 🌟 概述

LiziEngine2D 是一个二维静电场-粒子相互作用的物理模拟引擎，采用 **PIC (Particle-In-Cell)** 方法将离散的带电粒子散射到网格上，求解 Poisson 方程得到电势，再通过插值将电场力回传到粒子，从而实现高效的带电粒子系统仿真。

项目提供基于 **egui** 的实时图形界面，支持鼠标交互拖拽粒子、切换预设场景、实时显示电势热力图，直观展示静电场的物理行为。

---

## ✨ 功能特性

- **PIC 流水线实现**
  - 双线性散射（Particles → Grid 电荷密度 ρ）
  - 周期边界下 Poisson 方程 FFT 求解 → 电势 V
  - 中心差分计算电场 E = −∇V
  - 双线性 Gather（Grid 电场 → Particles 受力）
  - 半隐式欧拉积分更新粒子运动
- **实时 GUI 交互**
  - 电势热力图实时渲染（伪彩色映射）
  - 粒子位置实时显示（正电荷白色 / 负电荷青色）
  - 鼠标拖拽粒子（支持任意场景）
- **控制面板**
  - ▶️ Play / ⏸️ Pause / ⏭️ Step（单步执行）/ ⟳ Reset（重置）
  - 实时显示步数、电势范围、粒子数量
- **多种预设场景**
  - 单点电荷、双电荷同号、双电荷异号（偶极子）、随机粒子群
- **中文字体支持**
  - 自动检测系统字体，支持中文界面显示

---

## 🚀 快速开始

### 环境要求

- Rust 1.70+
- Cargo（Rust 包管理器）

### 构建与运行

```bash
# 克隆仓库
git clone https://github.com/TheSingularityStudio/LiziEngine2D.git
cd LiziEngine2D

# 构建（Release 模式以获得最佳性能）
cargo build --release

# 运行 GUI 模拟器（默认启动预设选择界面）
cargo run --release
```

### 命令行参数

```bash
# 指定窗口尺寸启动 GUI
cargo run --release -- gui --width 1024 --height 768
```

> 运行后默认显示预设选择界面，选择一个场景即可进入实时仿真。

---

## 🎮 预设场景

| 场景 | 描述 | 物理意义 |
|------|------|----------|
| **单点电荷** | 网格中央放置一个单位正电荷 | 显示静电场 V 热力图，观察点电荷电势分布 |
| **双电荷（同号）** | 左右两侧放置两个同号正电荷 | 观察叠加电场，两正电荷相互排斥 |
| **双电荷（异号）** | 左右两侧放置一正一负电荷 | 偶极子电场，正负电荷相互吸引形成偶极场 |
| **随机粒子** | 200 个随机初始化的粒子 | 展示 PIC 模拟动画，粒子在静电场中运动 |

### GUI 操作指南

1. **启动程序** → 进入预设选择界面
2. **点击预设按钮** → 进入对应场景的实时模拟
3. **控制模拟**：
   - `▶️ Play` — 开始自动步进模拟
   - `⏸️ Pause` — 暂停模拟
   - `⏭️ Step` — 单步执行（每点一次前进一帧）
   - `⟳ Reset` — 重置到初始状态
   - `← 返回` — 返回预设选择界面
4. **鼠标交互**：在画布上拖拽任意粒子，观察其对电场的影响

---

## 🏗️ 技术架构

### PIC 流水线流程

```
┌─────────────────────────────────────────────────────┐
│                  时间步循环 (dt)                      │
│                                                      │
│  粒子位置 ──→ Scatter(双线性) ──→ 网格电荷密度 ρ      │
│                                        ↓              │
│  粒子受力 ←── Gather(双线性) ←── 电场 E = −∇V       │
│                                        ↑              │
│  半隐式欧拉积分                           Poisson 求解(FFT) │
│                                        ↑              │
│  边界条件 / 速度限制                  电势 V            │
└─────────────────────────────────────────────────────┘
```

### 模块结构

```
src/
├── main.rs           # CLI 入口（clap 解析）
├── lib.rs            # 库入口，公开模块
├── core/             # 核心物理引擎
│   ├── grid.rs       # 网格定义
│   ├── particles.rs  # 粒子状态管理
│   ├── scatter.rs    # 电荷散射到网格
│   ├── poisson_fft.rs / poisson_solver.rs  # FFT Poisson 求解
│   ├── interp.rs     # 电场插值（Gather）
│   ├── integrator.rs # 半隐式欧拉积分器
│   ├── boundary.rs   # 边界条件 & 速度限制
│   └── sim.rs        # 模拟器主控（ElectrostaticSim2D）
├── gui/              # 图形界面
│   ├── app.rs        # egui 应用主体
│   └── interaction.rs # 鼠标交互状态
├── presets/          # 预设场景
│   └── config.rs     # 预设定义与构建
└── visual/           # 可视化工具
    └── colors.rs     # 热力图颜色映射
```

### 核心依赖

| 依赖 | 用途 |
|------|------|
| [ndarray](https://crates.io/crates/ndarray) | 多维数组操作（网格数据、粒子数据） |
| [ndrustfft](https://crates.io/crates/ndrustfft) | FFT Poisson 求解器 |
| [eframe/egui](https://crates.io/crates/eframe) | 即时模式 GUI 框架 |
| [clap](https://crates.io/crates/clap) | 命令行参数解析 |
| [rand](https://crates.io/crates/rand) | 随机粒子初始化 |

---

## 🛠️ 项目构建

```bash
# 调试构建
cargo build

# 发布构建（推荐用于实际运行）
cargo build --release

# 运行测试
cargo test

# 运行验证测试（验证物理正确性）
cargo test --test validation
```

---

## 📜 许可证

本项目基于 [MIT License](LICENSE) 开源。

---

## 🔗 相关链接

- [GitHub 仓库](https://github.com/TheSingularityStudio/LiziEngine2D)
- [PIC 方法介绍](https://en.wikipedia.org/wiki/Particle-in-cell)
- [egui 框架文档](https://docs.rs/egui/)

---

*用 Rust 写就的粒子模拟器，探索静电场的奇妙世界。*