---
layout: post
title: "FAST-LIVO2 源码全流程解析与文件导读"
date: 2026-01-21
categories: [学习, SLAM, 雷达SLAM, FAST-LVIO]
---

# FAST-LIVO2 源码全流程解析与文件导读

## 1. 核心摘要
> FAST-LIVO2 是一个**紧耦合**的激光雷达-惯性-视觉里程计系统。其核心逻辑在于：利用 **IMU 进行高频状态预测**（去畸变），然后交替使用 **视觉观测（稀疏光流/直接法）** 和 **雷达观测（体素平面匹配）** 对状态进行 **IESKF（迭代误差状态卡尔曼滤波）更新**。系统由 `LIVMapper` 类统一调度。

## 2. 关键概念/术语
- **LIVMapper**：系统的总指挥，负责数据同步和调度 LIO/VIO 模块。
- **IESKF (Iterated Error State Kalman Filter)**：后端优化的核心算法，无论是视觉还是雷达，最终都转化为 $H \delta x = r$ 的形式更新状态。
- **Voxel Map**：雷达部分的地图管理，使用哈希表维护体素内的平面特征。
- **Direct Method (直接法)**：视觉部分不计算特征描述子，而是直接利用光度误差（Photometric Error）进行对齐。

## 3. 详细内容整理

### 3.1 `src` 文件夹各文件功能解析

| 文件名 | 核心功能 | 详细说明 |
| :--- | :--- | :--- |
| **`main.cpp`** | **程序入口** | ROS 节点初始化。实例化 `LIVMapper` 对象，调用 `mapper.run()` 开启主循环。 |
| **`LIVMapper.cpp`** | **系统调度 (The Brain)** | 1. **订阅/发布**：接收 LiDAR、IMU、Image 消息。<br>2. **数据同步**：`sync_packages()` 函数负责将异构传感器数据对齐。<br>3. **状态机**：决定当前帧执行 VIO 更新还是 LIO 更新 (`handleVIO` / `handleLIO`)。<br>4. **保存结果**：负责保存 PCD 地图和轨迹文件。 |
| **`preprocess.cpp`** | **雷达预处理** | 适配不同型号雷达（Avia, Velodyne, Ouster 等）。计算点的曲率（用于时间戳去畸变），并进行简单的滤波。 |
| **`IMU_Processing.cpp`** | **IMU 处理 (Prediction)** | 1. **IMU 初始化**：估计重力方向和零偏。<br>2. **前向传播**：利用 IMU 数据预测当前位姿（作为 EKF 的先验）。<br>3. **去畸变 (Undistort)**：利用预测的运动，将雷达点云统一到帧头或帧尾时刻。 |
| **`voxel_map.cpp`** | **LIO 后端 (LiDAR Update)** | 1. **地图管理**：维护 `voxel_map_` 哈希表。<br>2. **残差构建**：`BuildResidualListOMP` 并行计算点到平面的距离残差。<br>3. **EKF 更新**：`StateEstimation` 计算雅可比矩阵 $H$，执行卡尔曼滤波更新状态。<br>4. **地图维护**：`UpdateVoxelMap` 插入新点，`mapSliding` 移除旧区域。 |
| **`vio.cpp`** | **VIO 后端 (Visual Update)** | 1. **视觉前端**：管理特征点提取与光流追踪（直接法）。<br>2. **视觉地图**：维护 `sub_feat_map`（稀疏特征地图）。<br>3. **EKF 更新**：`updateState` 计算光度误差残差和雅可比，更新状态。<br>4. **深度关联**：将雷达点云投影到图像上，赋予视觉特征深度值。 |
| **`frame.cpp`** | **视觉帧数据结构** | 存储图像金字塔、相机模型指针等单帧信息。 |
| **`visual_point.cpp`** | **视觉地图点** | 存储 3D 视觉点的信息（位置、法向量、观测到的特征块引用），用于构建视觉残差。 |

### 3.2 整体代码运行流程 (Pipeline)

系统运行在 `LIVMapper::run()` 的主循环中，整体流程如下：

#### 第一阶段：数据接收与同步
1.  **Callback**：`LIVMapper` 中的 `livox_pcl_cbk`, `imu_cbk`, `img_cbk` 将数据存入各自的缓冲队列。
2.  **Sync (`sync_packages`)**：
    * 从队列中取出数据。
    * 根据时间戳对齐 LiDAR、IMU 和 Image。
    * 确定当前帧是执行 **LIO**（只有雷达+IMU）还是 **VIO**（有图像插入）模式。
    * **关键点**：FAST-LIVO2 通常以图像帧率触发 VIO 更新，以雷达帧率触发 LIO 更新，两者交替进行。

#### 第二阶段：IMU 预测与去畸变
1.  调用 `processImu()` -> `p_imu->Process2()`。
2.  **前向传播**：利用上一帧状态和当前 IMU 数据，积分得到当前时刻的 **预测位姿 (Predict Pose)**。
3.  **点云去畸变**：利用 IMU 计算出的运动，消除雷达点云因自身运动产生的畸变 (`UndistortPcl`)。

#### 第三阶段：状态估计与建图 (State Estimation and Mapping)
根据 `sync_packages` 确定的标志位 (`lio_vio_flg`) 进入不同分支：

**分支 A：VIO 处理 (`handleVIO`)**
1.  **处理图像**：`vio_manager->processFrame()`。
2.  **特征检索**：从全局 Voxel Map 中检索当前视锥内的特征点 (`retrieveFromVisualSparseMap`)。
3.  **IESKF 更新**：
    * 计算图像光度误差（当前帧图像块 vs 参考帧图像块）。
    * 构建雅可比矩阵 $H$。
    * 更新状态向量（位置、姿态、速度、零偏等）。
4.  **特征点更新**：将当前帧中质量好的特征点加入视觉地图。

**分支 B：LIO 处理 (`handleLIO`)**
1.  **降采样**：对去畸变后的点云进行体素降采样。
2.  **构建残差**：`voxelmap_manager->BuildResidualListOMP()`。
    * 在 Voxel Map 中查找每个点所在的体素。
    * 拟合体素内的平面，计算 **点到平面距离** 作为残差。
3.  **IESKF 更新**：`voxelmap_manager->StateEstimation()`。
    * 计算点面残差的雅可比。
    * 更新状态向量（与 VIO 共享同一个状态向量）。
4.  **地图更新**：
    * `UpdateVoxelMap`：将当前帧的点云插入 Voxel Map，更新体素内的平面参数。
    * `mapSliding`：如果地图过大，移除距离当前位置过远的体素。

#### 第四阶段：发布与可视化
1.  发布里程计 (`pubOdomAftMapped`)。
2.  发布当前帧点云、轨迹路径。
3.  如果开启了保存功能，将点云写入 PCD 文件。

## 4. 核心源码深度学习

### 4.1 多传感器时间同步 (Sync Packages)
**目标**：理解系统是如何将高频雷达、IMU 和低频图像在时间轴上对齐，实现紧耦合的。
**文件**：`src/LIVMapper.cpp`

```cpp
bool LIVMapper::sync_packages(LidarMeasureGroup &meas)
{
  // 1. 基础检查：确保所有传感器的缓冲区都有数据
  if (lid_raw_data_buffer.empty() && lidar_en) return false;
  // ... (省略部分检查代码)

  switch (slam_mode_)
  {
  case LIVO: // 重点关注 LIVO 模式
  {
    /* 逻辑说明：通过 meas.lio_vio_flg 状态机控制交替更新。
       LIO 和 VIO 的更新通常是对齐到图像时刻的。 */
    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;

    switch (last_lio_vio_flg)
    {
    // case WAIT/VIO: 意味着上一次是 VIO 更新或刚开始，现在准备处理雷达数据
    case WAIT:
    case VIO:
    {
      // [关键] 获取当前图像帧的拍摄时间，这是本次同步的锚点
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      
      // ... (省略时间戳异常检查)

      // [切割] 将雷达扫描分割。只取 scan_start 到 img_capture_time 这一段数据
      struct MeasureGroup m;
      m.lio_time = img_capture_time; // 本次 LIO 更新的目标时间点设为图像时间
      
      // 提取对应时段的 IMU 数据
      // ... (省略 IMU 提取循环)

      // [核心算法] 遍历雷达缓冲区，进行点云切割
      // 将跨越 img_capture_time 的雷达帧一分为二：
      // Part A: 用于本次 LIO 更新 (meas.pcl_proc_cur)
      // Part B: 留给下一次 (meas.pcl_proc_next)
      while (!lid_raw_data_buffer.empty())
      {
        if (lid_header_time_buffer.front() > img_capture_time) break; 
        
        auto pcl(lid_raw_data_buffer.front()->points);
        double frame_header_time(lid_header_time_buffer.front());
        // 计算切割的时间阈值（转为毫秒）
        float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

        for (int i = 0; i < pcl.size(); i++)
        {
          auto pt = pcl[i];
          if (pcl[i].curvature < max_offs_time_ms) {
            // 时间戳 < 图像时间，归入当前帧处理
            pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
            meas.pcl_proc_cur->points.push_back(pt);
          } else {
            // 时间戳 > 图像时间，归入下一帧缓存
            pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
            meas.pcl_proc_next->points.push_back(pt);
          }
        }
        // ...
      }

      meas.lio_vio_flg = LIO; // 标记状态为 LIO，指示系统下一步执行 StateEstimation
      return true;
    }

    case LIO: // 如果刚刚完成了 LIO 更新，紧接着准备 VIO 更新
    {
      double img_capture_time = img_time_buffer.front() + exposure_time_init;
      meas.lio_vio_flg = VIO; // 切换状态为 VIO
      
      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.img = img_buffer.front(); // 取出对应的图像
      
      // 弹出已处理的图像，准备处理下一帧
      img_buffer.pop_front();
      return true;
    }
    }
    break;
  }
  }
  return false;
}
```

### 4.2 LIO 状态估计 (LiDAR State Estimation)

**核心任务**：利用 **点到平面距离** 构建残差，通过 **IESKF** 迭代更新状态。
**文件位置**：`src/voxel_map.cpp`
**关键函数**：`VoxelMapManager::StateEstimation`

该函数在每一帧雷达数据处理时被调用。它使用当前预测的状态（来自 IMU 积分）作为初值，通过多次迭代优化，使雷达点云与地图平面的贴合度最好。

```cpp
void VoxelMapManager::StateEstimation(StatesGroup &state_propagat)
{
  // 清空用于存储中间计算结果的容器
  cross_mat_list_.clear(); 
  body_cov_list_.clear(); 
  // ... (省略部分预处理代码)

  // [IESKF 核心循环] 默认最大迭代 5 次 (config_setting_.max_iterations_)
  // 每一次迭代都会利用更新后的状态重新寻找匹配和计算雅可比，以处理非线性问题
  for (int iterCount = 0; iterCount < config_setting_.max_iterations_; iterCount++)
  {
    // 1. 将降采样后的点云转换到世界坐标系
    // 使用当前迭代的状态估计值 (state_.rot_end, state_.pos_end)
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>);
    TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, world_lidar);
    
    // ... (更新每个点的方差信息，考虑姿态不确定性的传播) ...

    // 2. [核心步骤] 并行构建残差列表
    // 函数内部会遍历每个点，在 VoxelMap 哈希表中查找最近的体素和平面
    // 如果找到匹配平面，计算点到平面的距离 (dis_to_plane_) 和法向量 (normal_)
    // 结果存放在 ptpl_list_ (Point-To-Plane List) 中
    BuildResidualListOMP(pv_list_, ptpl_list_);

    // 统计有效特征点数量 (成功匹配到平面的点)
    effct_feat_num_ = ptpl_list_.size();

    // 3. 构建观测雅可比矩阵 H 和残差向量
    // Hsub: 观测矩阵 (N x 6)，N 为有效点数，6 为状态维度 (旋转3 + 平移3)
    // meas_vec: 观测残差向量 z = 0 - h(x)
    MatrixXd Hsub(effct_feat_num_, 6);
    MatrixXd Hsub_T_R_inv(6, effct_feat_num_); // H^T * R^-1，用于简化后续计算
    VectorXd R_inv(effct_feat_num_);           // 观测噪声协方差 R 的逆
    VectorXd meas_vec(effct_feat_num_);

    // 遍历所有有效匹配点，填充 H 矩阵
    for (int i = 0; i < effct_feat_num_; i++)
    {
      auto &ptpl = ptpl_list_[i]; // 获取第 i 个点的匹配信息
      
      // 计算点在世界系下的坐标的反对称矩阵 A
      // 公式推导：对点 P_w = R * P_b + t 求关于 (delta_theta, delta_t) 的导数
      // d(P_w)/d(delta_theta) = -(R * P_b + t)^ (反对称矩阵)
      V3D point_this(ptpl.point_b_);
      point_this = extR_ * point_this + extT_; // 转换到 IMU 坐标系
      V3D A(point_crossmat * state_.rot_end.transpose() * ptpl_list_[i].normal_);
      
      // 填充 H 矩阵的一行
      // 前 3 列：对旋转的雅可比 (A)
      // 后 3 列：对平移的雅可比 (即平面的法向量 n)
      Hsub.row(i) << VEC_FROM_ARRAY(A), ptpl_list_[i].normal_[0], ptpl_list_[i].normal_[1], ptpl_list_[i].normal_[2];
      
      // 填充观测残差
      // 目标是让点到平面距离为 0，所以残差 z = 0 - dist = -dis_to_plane_
      meas_vec(i) = -ptpl_list_[i].dis_to_plane_;
      
      // 计算观测权重 R_inv
      // sigma_l 是平面的几何不确定性，var 是点本身的测量噪声
      // 距离越远、平面拟合越差，R_inv 越小（权重越低）
      double sigma_l = J_nq * ptpl_list_[i].plane_var_ * J_nq.transpose();
      R_inv(i) = 1.0 / (0.001 + sigma_l + ptpl_list_[i].normal_.transpose() * var * ptpl_list_[i].normal_);
      
      // 预计算 H^T * R^-1 的一部分，便于后续矩阵乘法
      Hsub_T_R_inv.col(i) << A[0] * R_inv(i), ... , ptpl_list_[i].normal_[2] * R_inv(i);
    }

    // 4. IESKF 更新步骤
    // 标准卡尔曼增益公式: K = P * H^T * (H * P * H^T + R)^-1
    // 使用信息矩阵形式（Sherman-Morrison-Woodbury 变换）加速计算:
    // K = (H^T * R^-1 * H + P^-1)^-1 * H^T * R^-1
    
    // 计算 H^T * R^-1 * H (Hessian 矩阵的近似)
    H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub; 
    
    // 计算 (H^T * R^-1 * H + P^-1)^-1
    // state_.cov 是上一时刻的后验协方差 P，这里取逆得到信息矩阵 P^-1
    MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H.block<DIM_STATE, DIM_STATE>(0, 0) + state_.cov.block<DIM_STATE, DIM_STATE>(0, 0).inverse()).inverse();
    
    // 计算等效卡尔曼增益项 G = K * H
    G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
    
    // 计算状态更新量 solution (delta_x)
    // vec = x_pred - x_k (预测状态 - 当前迭代状态)
    // solution = K * (z - H * dx) ... 这里包含了一些对 IESKF 先验项的处理
    auto vec = state_propagat - state_;
    VD(DIM_STATE) solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec.block<DIM_STATE, 1>(0, 0) - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
    
    // 更新状态向量
    // state_ 是当前迭代的估计值 x_k
    state_ += solution;

    // 5. 判断收敛
    // 如果旋转更新量 < 0.01 度 且 平移更新量 < 1.5 cm，则认为收敛
    auto rot_add = solution.block<3, 1>(0, 0);
    auto t_add = solution.block<3, 1>(3, 0);
    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015)) 
    { 
        flg_EKF_converged = true; 
    }
    
    // ... (收敛后更新后验协方差 P 并退出循环) ...
  }
}
```

## 4.3 VIO 状态估计 (Visual State Estimation)

**核心任务**：利用 **直接法 (Direct Method)** 计算光度误差，构建雅可比矩阵，更新系统状态（位姿 + 曝光时间）。
**文件位置**：`src/vio.cpp`
**关键函数**：`VIOManager::updateState`

VIO 部分同样采用 IESKF 框架。与 LIO 不同的是，它的残差来源是**像素灰度值的差异**。FAST-LIVO2 采用了稀疏直接法，即只对提取出的特征点周围的 patch（小图像块，如 4x4 或 5x5）进行光度误差计算，而不是整张图像。

```cpp
void VIOManager::updateState(cv::Mat img, int level)
{
  if (total_points == 0) return; // 如果没有特征点，直接返回
  StatesGroup old_state = (*state); // 备份旧状态，用于迭代失败回滚

  // 定义 IESKF 所需矩阵
  VectorXd z;      // 残差向量
  MatrixXd H_sub;  // 雅可比矩阵
  
  // 预分配内存：总维度 = 特征点数 * 每个特征点的 Patch 像素数
  const int H_DIM = total_points * patch_size_total;
  z.resize(H_DIM);
  z.setZero();
  // 状态维度：6 (Pose) + 1 (Exposure Time) = 7
  H_sub.resize(H_DIM, 7); 
  H_sub.setZero();

  // [外部循环] IESKF 迭代优化，默认 max_iterations 次
  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();

    // 1. 准备当前迭代的状态量
    M3D Rwi(state->rot_end);   // 世界 -> IMU 旋转
    V3D Pwi(state->pos_end);   // 世界 -> IMU 平移
    
    // 计算 世界 -> 相机 的变换 T_cw = T_ci * T_iw
    // Rcw = Rci * Rwi^T
    Rcw = Rci * Rwi.transpose(); 
    // Pcw = -Rci * Rwi^T * Pwi + Pci
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    
    // 预计算一些雅可比的公共部分
    // Jdp_dt = d(P_c)/d(t_wi) = R_ci * R_wi^T
    Jdp_dt = Rci * Rwi.transpose(); 
    
    float error = 0.0;
    int n_meas = 0;
  
    // [OpenMP 并行加速] 遍历所有特征点，计算光度残差和雅可比
    #ifdef MP_EN
      omp_set_num_threads(MP_PROC_NUM);
      #pragma omp parallel for reduction(+:error, n_meas)
    #endif
    for (int i = 0; i < total_points; i++)
    {
      MD(1, 2) Jimg;   // 图像梯度
      MD(2, 3) Jdpi;   // 投影导数
      MD(1, 3) Jdphi, Jdp, JdR, Jdt; // 链式法则中间变量

      float patch_error = 0.0;
      int search_level = visual_submap->search_levels[i];
      int pyramid_level = level + search_level;
      int scale = (1 << pyramid_level); // 当前金字塔层级的缩放倍率
      float inv_scale = 1.0f / scale;

      VisualPoint *pt = visual_submap->voxel_points[i];
      if (pt == nullptr) continue;

      // 2. 将特征点从世界系投影到当前帧图像平面
      // P_c = R_cw * P_w + t_cw
      V3D pf = Rcw * pt->pos_ + Pcw;
      // p_uv = K * P_c / Z
      V2D pc = cam->world2cam(pf);

      // 计算投影函数的雅可比 J_dpi = d(u,v)/d(P_c)
      // J_dpi = [fx/Z, 0, -fx*X/Z^2; 
      //          0, fy/Z, -fy*Y/Z^2]
      computeProjectionJacobian(pf, Jdpi);
      
      // p_hat 是 P_c 的反对称矩阵，用于计算旋转导数
      M3D p_hat;
      p_hat << SKEW_SYM_MATRX(pf);

      // 计算参考帧中的像素坐标 (sub-pixel precision)
      // ... (双线性插值系数计算代码省略) ...

      // 获取参考帧的 Patch 像素值 (P) 和参考曝光时间 (inv_ref_expo)
      vector<float> P = visual_submap->warp_patch[i];
      double inv_ref_expo = visual_submap->inv_expo_list[i];

      // 3. 遍历 Patch 内的所有像素 (例如 4x4)
      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = ...; // 指向当前像素数据
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          // 计算图像梯度 (du, dv) = d(I)/d(u,v)
          float du = ...; 
          float dv = ...;

          Jimg << du, dv;
          Jimg = Jimg * state->inv_expo_time; // 乘以曝光时间系数
          Jimg = Jimg * inv_scale;            // 考虑金字塔缩放

          // 4. 链式法则计算雅可比
          // d(I)/d(R) = d(I)/duv * d(uv)/dP_c * d(P_c)/d(R)
          // d(P_c)/d(R) ~ -[P_c]_x
          Jdphi = Jimg * Jdpi * p_hat;
          
          // d(I)/d(t) = d(I)/duv * d(uv)/dP_c * d(P_c)/d(t)
          // d(P_c)/d(t) = I (这里通过 Jdp_dt 处理了外参)
          Jdp = -Jimg * Jdpi; // 注意这里的符号定义
          
          // 转换到对世界系状态 (R_wi, t_wi) 的导数
          JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
          Jdt = Jdp * Jdp_dt;

          // 获取当前帧像素值 (双线性插值)
          double cur_value = ...;
          
          // 5. 计算光度误差残差
          // 模型：I_cur * dt_cur = I_ref * dt_ref
          // 残差 r = I_cur * dt_cur - I_ref * dt_ref
          // 这里的 state->inv_expo_time 实质上是 1/dt (或者相关的曝光系数)
          double res = state->inv_expo_time * cur_value - inv_ref_expo * P[...];

          z(i * patch_size_total + x * patch_size + y) = res;
          patch_error += res * res;
          n_meas += 1;
          
          // 6. 填充雅可比矩阵 H_sub 的一行
          // 最后一列是对曝光时间系数的偏导：d(res)/d(inv_expo) = cur_value
          if (exposure_estimate_en) { 
              H_sub.block<1, 7>(...) << JdR, Jdt, cur_value; 
          } else { 
              H_sub.block<1, 6>(...) << JdR, Jdt; 
          }
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas; // 计算平均误差

    // 7. IESKF 状态更新
    // 只有当总误差下降时才执行更新 (类似 Levenberg-Marquardt 的接受准则)
    if (error <= last_error)
    {
      old_state = (*state); // 更新成功，保存当前状态为“最佳状态”
      last_error = error;

      // 构建 Hessian 近似 H^T * H
      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;
      
      // 计算卡尔曼增益 K 的中间量 (H^T * H + P^-1)^-1
      // img_point_cov 是视觉测量的噪声方差 R
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      
      auto &&HTz = H_sub_T * z; // H^T * z
      
      // 计算先验误差项: vec = x_pred - x_k
      auto vec = (*state_propagat) - (*state);
      
      // 计算等效增益矩阵 G = K * H (用于后续协方差更新)
      G.block<DIM_STATE, 7>(0, 0) = K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
      
      // 计算状态更新量 solution = K * (z_innov) + ...
      // 这里的公式形式是 IESKF 的标准推导结果
      MD(DIM_STATE, 1) solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec - G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);

      // 更新状态向量
      (*state) += solution;

      // 8. 判断收敛
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);
      
      // 如果旋转更新 < 0.001度 且 平移更新 < 0.001m，则认为收敛，提前退出循环
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))  EKF_end = true;
    }
    else
    {
      // 误差反而变大了，回滚到上一次的状态，并强制结束迭代
      (*state) = old_state;
      EKF_end = true;
    }

    if (iteration == max_iterations || EKF_end) break;
  }
}
```