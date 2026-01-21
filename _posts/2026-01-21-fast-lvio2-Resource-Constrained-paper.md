---
layout: post
title: "论文阅读：FAST-LIVO2 Resource - 资源受限平台上的高效 LIVO"
date: 2026-01-21
categories: [学习, SLAM, 雷达SLAM, FAST-LVIO]
---

## 1. 核心摘要
> 这篇笔记整理自 **FAST-LIVO2 on Resource-Constrained Platforms** (arXiv:2501.13876)。针对边缘计算设备（如 ARM 平台），作者对 FAST-LIVO2 进行了深度剪枝和优化。核心策略包括**退化感知的自适应帧选择器**和**序列化 ESIKF 更新**，在 Hilti 数据集上实现了 **33% 的速度提升**和 **47% 的内存减少**，且精度仅损失 3cm。

## 2. 关键概念/术语
- **Degeneration-aware Selector (退化感知选择器)**：一种机制，能够识别当前运动是否处于退化状态（如匀速直线运动或纯旋转），从而智能决定是否丢弃视觉帧以节省算力。
- **Sequential Update (序列化更新)**：在卡尔曼滤波中，不一次性处理所有测量值，而是分批次或按顺序更新状态，降低单次矩阵运算的维度。
- **Long-term Visual Map (长期视觉地图)**：为了平衡内存，将地图分为局部高精度的统一地图和全局的长期稀疏地图。

<div style="text-align: center;">
  <img src="/assets/images/FAST-LVIO/2026-01-21-fast-livo2-Resource-Constrained.png" style="width: 90%; height: auto; border-radius: 4px;">
  <p style="color: #666; font-size: 0.9rem; margin-top: 5px;">资源受限平台优化策略</p>
</div>

## 3. 详细内容整理

### 3.1 优化策略
- **计算优化**：
    - 引入**自适应视觉帧选择器**。如果系统检测到 LiDAR 状态估计良好且无退化风险，会主动跳过部分视觉帧的处理，减少 CPU 占用。
    - **ESIKF 序列更新**：将大规模的矩阵求逆操作拆解，降低峰值计算量。
- **内存优化**：
    - **双层地图结构**：
        1.  **局部统一视觉-LiDAR 地图**：用于当前的精确实时跟踪。
        2.  **长期视觉地图**：仅存储关键信息，用于闭环或重定位，大幅降低 RAM 占用。

### 3.2 实验结果
- **平台**：验证了 x86 和 ARM (嵌入式) 平台。
- **性能对比**：相比原版 FAST-LIVO2，运行时间减少 33%，内存占用减少 47%。
- **精度权衡**：在 Hilti 数据集上，RMSE 仅增加了 3cm，这对于大多数机器人导航任务是完全可接受的。

## 4. 关键知识点详解

### 4.1 序列化更新 (Sequential Update) 的数学原理
标准的 Kalman Filter 更新步骤涉及计算卡尔曼增益 $K$：
$$K = P H^T (H P H^T + R)^{-1}$$
当测量维度（$H$ 的行数）很大时（例如同时融合数百个 LiDAR 点和视觉像素），$(H P H^T + R)$ 的求逆非常耗时。

序列化更新利用了测量值之间的独立性假设，将观测 $z$ 分为 $z_1, z_2, ... z_n$。
- 先用 $z_1$ 更新状态 $x$ 和协方差 $P$。
- 得到新的 $x', P'$ 后，再用 $z_2$ 进行更新。
- 这样避免了构建巨大的观测矩阵，显著降低了计算复杂度，特别适合算力有限的嵌入式 CPU。