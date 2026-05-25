# Vascular_Statistics

三维血管网络形态学统计后处理工具。

## 功能

- **骨架化**（VascGraph）：从 3D 二值分割图像自动提取血管中轴图（Laplacian Flow Dynamics 算法）
- **3D 交互可视化**：mayavi 图形界面，支持半径/类型/流速/压力编码
- **形态学统计**（C++）：直径、路径长度、弯曲度（tortuosity）、段密度
- **13 种格式读写**：Pajek、SWC、MAT、CGAL、SOAX 等

## 快速开始

```bash
# 安装
pip install -e .

# 完整管线
vascular-stats pipeline input.mat -v 0.078

# 仅骨架化
vascular-stats skeletonize input.mat -o graph.pajek

# 仅统计
vascular-stats stats graph -v 0.078
```

## 依赖

- Python ≥ 3.8, numpy, scipy, networkx ≥ 2.2, scikit-image, matplotlib, click, h5py
- C++17 编译器（g++ 或 MSVC）
- 可选 GUI：mayavi, traits, traitsui（推荐 conda-forge 安装）

## 目录结构

```
Vascular_Statistics/
├── include/vascular_statistics/   # C++ 头文件
├── src/                           # C++ 源码
├── python/vascular_statistics/    # Python 包
│   ├── vascgraph/VascGraph/       #   VascGraph 骨架化与可视化
│   ├── bridge.py                  #   格式桥
│   └── cli.py                     #   CLI 入口
├── data/samples/                  # 样例数据
├── CMakeLists.txt / pyproject.toml
└── docs/
```

## 许可

- C++ 统计模块：本仓库
- VascGraph 子模块：MIT License (c) Rafat Damseh
