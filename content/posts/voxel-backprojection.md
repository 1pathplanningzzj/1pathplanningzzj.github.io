---
title: "体素反向投影：从 2D 图像特征到 3D Gaussian 表征"
date: 2026-01-19
draft: false
math: true
tags: ["3D视觉", "体素", "多视图融合", "DF3DGS"]
---

## 核心思想

在 3D 场景理解中，我们需要将多个 2D 图像的特征融合成统一的 3D 表征。DF3DGS 采用了一种高效的**体素反向投影 (Voxel Back-projection)** 机制，也称为**反向扭曲 (Inverse Warping)**。

与传统的"先生成点云再融合"不同，这种方法：
1. **预定义** 3D 空间中的体素网格
2. **投影** 体素点到各个相机的 2D 图像平面
3. **采样** 对应像素位置的特征
4. **填充** 3D 体素空间
5. **融合** 多视图特征

这个过程的关键在于：**不是从 2D 推 3D，而是从 3D 查询 2D**。

## 完整流程详解

### 步骤 1: 预定义世界坐标系下的 3D 体素网格

```python
# volumetric_fusionnet.py, 第 116 行
v_pts_local = torch.matmul(ext_inv_mat, self.voxel_pts.repeat(bs, 1, 1))
```

**关键概念**：`self.voxel_pts` 是预先定义的 3D 空间体素点集合。

这些点形成一个规则的 3D 网格，覆盖感兴趣的空间区域。例如：
- 空间范围：$[-1, 1] \times [-1, 1] \times [0, 2]$ 米
- 分辨率：$64 \times 64 \times 32$ 个体素
- 每个体素代表一个候选的 3D 位置

**数学表示**：
$$
\mathbf{P}_{\text{world}} = \{\mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_N\} \in \mathbb{R}^{3 \times N}
$$

其中 $N$ 是体素总数（如 $64 \times 64 \times 32 = 131072$）。

### 步骤 2: 世界坐标系 → 相机坐标系（外参变换）

```python
# volumetric_fusionnet.py, 第 115-117 行
ext_inv_mat = torch.inverse(ext_mat)  # 外参逆矩阵
v_pts_local = torch.matmul(ext_inv_mat, self.voxel_pts.repeat(bs, 1, 1))
```

**作用**：将世界坐标系下的体素点转换到当前相机的局部坐标系。

**数学表示**：
$$
\mathbf{P}_{\text{camera}} = \mathbf{R}^T (\mathbf{P}_{\text{world}} - \mathbf{t})
$$

其中：
- $\mathbf{R} \in \mathbb{R}^{3 \times 3}$：相机旋转矩阵
- $\mathbf{t} \in \mathbb{R}^3$：相机平移向量
- 外参矩阵：$\mathbf{T} = [\mathbf{R} \mid \mathbf{t}]$

**齐次坐标形式**：
$$
\begin{bmatrix} \mathbf{P}_{\text{camera}} \\ 1 \end{bmatrix} =
\begin{bmatrix} \mathbf{R}^T & -\mathbf{R}^T \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}
\begin{bmatrix} \mathbf{P}_{\text{world}} \\ 1 \end{bmatrix}
$$

**物理意义**：
- 对于 Agent 相机：体素点从世界坐标系转到 Agent 视角
- 对于 Wrist 相机：体素点从世界坐标系转到 Wrist 视角
- 不同相机看到的是同一个 3D 点的不同视角

### 步骤 3: 相机坐标系 → 像素坐标系（内参投影）

```python
# volumetric_fusionnet.py, 第 120-121 行
# calculate_sample_pixel_coords 函数（第 148 行）
pix_coords = calculate_sample_pixel_coords(v_pts_local, int_mat)
```

**作用**：利用相机内参，将 3D 点投影到 2D 图像平面。

**数学表示**：
$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim
\mathbf{K} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
$$

其中内参矩阵：
$$
\mathbf{K} = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

**展开形式**：
$$
u = \frac{f_x \cdot X_c}{Z_c} + c_x, \quad v = \frac{f_y \cdot Y_c}{Z_c} + c_y
$$

**参数含义**：
- $f_x, f_y$：焦距（像素单位）
- $c_x, c_y$：主点坐标（图像中心）
- $(u, v)$：像素坐标
- $(X_c, Y_c, Z_c)$：相机坐标系下的 3D 点

**关键性**：内参的准确性直接决定了投影的正确性。如果 $f_x, f_y$ 不准确，体素点会投影到错误的像素位置，采样到错误的特征。

### 步骤 4: 特征采样（从 2D 到 3D）

```python
# volumetric_fusionnet.py, 第 126 行
sampled_feat = F.grid_sample(
    feats_img,
    pix_coords,
    mode='bilinear',
    padding_mode='zeros',
    align_corners=True
)
```

**作用**：根据计算出的像素坐标，从 2D 特征图中采样特征，填充到 3D 体素中。

**过程**：
1. 输入：2D 特征图 $\mathbf{F}_{\text{2D}} \in \mathbb{R}^{C \times H \times W}$
2. 查询：像素坐标 $(u, v)$
3. 采样：双线性插值获取特征向量 $\mathbf{f} \in \mathbb{R}^C$
4. 填充：将特征 $\mathbf{f}$ 赋值给对应的 3D 体素

**数学表示**：
$$
\mathbf{V}(\mathbf{p}_i) = \text{Sample}(\mathbf{F}_{\text{2D}}, \pi(\mathbf{K}, \mathbf{R}, \mathbf{t}, \mathbf{p}_i))
$$

其中：
- $\mathbf{V}(\mathbf{p}_i)$：体素 $i$ 的特征
- $\pi(\cdot)$：投影函数（步骤 2 + 步骤 3）
- $\text{Sample}(\cdot)$：双线性插值采样

**结果**：原本"扁平"的 2D 特征图被"立体化"，填充到 3D 空间中。

### 步骤 5: 多视图融合

```python
# volumetric_fusionnet.py, 第 135 行
voxel_feat_list.append(sampled_feat)
# 后续的 preprocess_overlap 进行融合
```

**作用**：将来自不同相机视角的特征进行加权融合。

**融合策略**：

1. **简单平均**：
$$
\mathbf{V}_{\text{fused}}(\mathbf{p}) = \frac{1}{M} \sum_{m=1}^{M} \mathbf{V}_m(\mathbf{p})
$$

2. **加权融合**（考虑可见性）：
$$
\mathbf{V}_{\text{fused}}(\mathbf{p}) = \frac{\sum_{m=1}^{M} w_m(\mathbf{p}) \cdot \mathbf{V}_m(\mathbf{p})}{\sum_{m=1}^{M} w_m(\mathbf{p})}
$$

其中权重 $w_m(\mathbf{p})$ 可以基于：
- 视角角度（正面视角权重更高）
- 深度置信度
- 特征一致性

**物理意义**：
- Agent 相机看到物体正面 → 提供正面特征
- Wrist 相机看到物体侧面 → 提供侧面特征
- 融合后的 3D 体素包含完整的多视角信息

## 完整的数学流程

对于体素点 $\mathbf{p}_i \in \mathbb{R}^3$，从世界坐标系到最终的 3D 特征：

$$
\mathbf{V}_{\text{fused}}(\mathbf{p}_i) = \text{Fuse}\left(\left\{\text{Sample}\left(\mathbf{F}_m, \pi_m(\mathbf{p}_i)\right)\right\}_{m=1}^{M}\right)
$$

其中投影函数：
$$
\pi_m(\mathbf{p}_i) = \mathbf{K}_m \cdot \mathbf{R}_m^T (\mathbf{p}_i - \mathbf{t}_m)
$$

展开为：
$$
\begin{bmatrix} u_m \\ v_m \\ 1 \end{bmatrix} \sim
\begin{bmatrix}
f_{x,m} & 0 & c_{x,m} \\
0 & f_{y,m} & c_{y,m} \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
\mathbf{R}_m^T & -\mathbf{R}_m^T \mathbf{t}_m
\end{bmatrix}
\begin{bmatrix} \mathbf{p}_i \\ 1 \end{bmatrix}
$$

## 为什么这种方法高效？

### 1. 避免显式点云生成

传统方法：
```
2D 图像 → 深度估计 → 点云生成 → 点云融合 → 体素化
```

体素反向投影：
```
预定义体素 → 投影查询 → 特征采样 → 直接融合
```

**优势**：
- 无需深度估计（不确定性大）
- 无需点云处理（计算密集）
- 直接在规则网格上操作（GPU 友好）

### 2. 规则的数据结构

体素网格是规则的 3D 数组，支持：
- 高效的并行计算
- 简单的索引访问
- 直接输入到 3D CNN

### 3. 可微分的端到端训练

整个流程完全可微：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{F}_{\text{2D}}} = \frac{\partial \mathcal{L}}{\partial \mathbf{V}_{\text{fused}}} \cdot \frac{\partial \mathbf{V}_{\text{fused}}}{\partial \mathbf{F}_{\text{2D}}}
$$

梯度可以从 3D 损失反向传播到 2D 特征提取器。

## 内外参的关键作用

### 内参 $\mathbf{K}$ 的作用

**决定投影的准确性**：
- 如果 $f_x, f_y$ 过大：体素点投影到图像边缘外，采样失败
- 如果 $f_x, f_y$ 过小：体素点投影到错误的像素，采样错误特征
- 如果 $c_x, c_y$ 偏移：整体投影位置偏移

**示例**：
```python
# 正确的内参
K_correct = [[500, 0, 320],
             [0, 500, 240],
             [0, 0, 1]]

# 错误的内参（焦距过大）
K_wrong = [[1000, 0, 320],  # 焦距翻倍
           [0, 1000, 240],
           [0, 0, 1]]
# 结果：体素点投影位置偏移 2 倍，采样到完全错误的特征
```

### 外参 $\mathbf{R}, \mathbf{t}$ 的作用

**决定多视图对齐**：
- 如果外参准确：不同相机的特征正确对齐到同一 3D 位置
- 如果外参缺失：所有相机的特征叠加到错误的位置，产生"鬼影"

**示例**：
```python
# Agent 相机外参
R_agent = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]
t_agent = [0, 0, 0]

# Wrist 相机外参（相对 Agent 偏移 0.2m）
R_wrist = [[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]
t_wrist = [0.2, 0, 0]

# 如果外参准确：
# - Agent 看到的物体正面 → 填充到体素 (x, y, z)
# - Wrist 看到的物体侧面 → 填充到体素 (x+0.2, y, z)
# - 两个视角的特征正确分布在 3D 空间

# 如果外参缺失（都用单位矩阵）：
# - Agent 看到的物体正面 → 填充到体素 (x, y, z)
# - Wrist 看到的物体侧面 → 也填充到体素 (x, y, z)
# - 两个视角的特征错误叠加，产生混乱
```

## 应用场景

### 1. 机器人操作（Embodied AI）

```python
# Agent 相机：全局视角，看到整个场景
# Wrist 相机：局部视角，看到操作细节
# 融合后：既有全局理解，又有局部精度
```

### 2. 自动驾驶

```python
# 多个车载相机：前、后、左、右
# 融合后：360° 的 3D 场景理解
```

### 3. 3D 重建

```python
# 多视角图像 → 3D 体素特征 → 3D Gaussian Splatting
# 用于高质量的场景重建和新视角合成
```

## 总结

体素反向投影是一种**从 3D 查询 2D** 的高效方法：

1. **预定义** 3D 体素网格（候选 3D 位置）
2. **外参变换** 将体素点转到各相机坐标系
3. **内参投影** 将 3D 点投影到 2D 像素
4. **特征采样** 从 2D 特征图采样填充 3D 体素
5. **多视图融合** 整合不同视角的信息

**关键要素**：
- **内参 $\mathbf{K}$**：保证投影准确，采样到正确的像素特征
- **外参 $\mathbf{R}, \mathbf{t}$**：保证多视图对齐，避免特征错位

这种方法将 2D 图像特征优雅地"立体化"为 3D Gaussian 表征，为后续的 3D 理解和生成任务提供了强大的基础。
