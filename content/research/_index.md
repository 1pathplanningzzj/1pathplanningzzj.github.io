---
title: "研究方向"
date: 2025-01-17
draft: false
hidemeta: true
---

## 研究领域

- 自动驾驶 矢量高清地图
- 2D 、3D 目标检测
- 硬件与嵌入式 机器人
- 具身智能 VLA (Vision-Language-Action)

## 当前项目

### BEVfusion 3D目标检测系统

**时间**: 2024.09 - 2025.01

基于 BEVfusion 框架实现多模态融合的3D目标检测系统。使用 Swin Transformer + FPN + Depth LSS 构建鸟瞰图（BEV）特征表示，结合稀疏点云编码器（SparseEncoder）进行多模态特征融合。在 NVIDIA Orin 平台上使用 TensorRT 进行模型优化，实现高效的实时3D目标检测。

**关键技术**: BEV感知、多模态融合、TensorRT优化、Orin平台部署

### 多车协同感知系统

**时间**: 2025.03 - 2025.06

研究车联网（V2V）环境下的多车协同感知技术，探索如何通过车辆间通信提升感知性能。基于 PointPillars 算法进行优化，在 NVIDIA Orin 平台上实现 10ms 的检测延迟。通过 V2V 通信协议和感兴趣区域（RoI）融合技术，实现多车协同目标检测，显著提升检测精度和鲁棒性。

**关键技术**: V2V通信、协同感知、PointPillars、RoI融合、边缘计算

### YOLOv8 目标检测优化

**时间**: 2025.01 - 2025.03

针对小目标检测场景对 YOLOv8 算法进行优化改进。在网络中添加 P2 特征层（4倍下采样）以增强小目标检测能力，在 Backbone 中融合注意力机制（SE/CA/ECA）+ C2f 模块提升特征提取能力。在 COCO 数据集上达到 67.2 mAP，模型参数量仅为 Co-DETR 的 1/10，达到 SOTA 水平的检测算法。

**关键技术**: YOLOv8、注意力机制、小目标检测、模型轻量化

### Autolabel 自动标注系统

**时间**: 2025.09 - 2025.12

开发基于 VMA（Vision-Language-Action）的自动标注系统，用于高效的数据标注。通过优化标注流程，集成预标注、主动学习、半监督学习等技术，实现标注效率提升7-8倍，大幅降低人工标注成本。

**关键技术**: 自动标注、主动学习、半监督学习、VMA

## 发表论文

1. **Software-Defined Parallel LiDARs for Active 3D Perception** - Zijian Zhang, Yuhang Liu, Boyi Sun, Jing Yang, Yutong Wang, Fei-Yue Wang, 2025 21st IEEE International Conference on Mechatronic and Embedded Systems and Applications (MESA), 2025. [[链接]](https://ieeexplore.ieee.org/document/11278878)
