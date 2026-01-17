---
title: "深度学习特征可视化与调试技巧"
date: 2025-01-17
draft: false
tags: ["深度学习", "特征可视化", "调试", "PyTorch"]
categories: ["知识库"]
---

在深度学习模型开发过程中，理解模型内部的特征表示和调试模型行为是至关重要的。本文将介绍常用的特征可视化方法和调试技巧。

## 为什么需要特征可视化？

- **理解模型学习内容**：查看模型在不同层学到了什么特征
- **诊断模型问题**：发现梯度消失、特征退化等问题
- **优化模型结构**：根据特征分布调整网络架构
- **提升模型可解释性**：让模型决策过程更透明

## 常用特征可视化方法

### 1. 特征图（Feature Maps）可视化

查看卷积层输出的特征图，了解网络在不同层提取的特征。

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_feature_maps(model, input_image, layer_name):
    """可视化指定层的特征图"""
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # 注册hook
    layer = dict(model.named_modules())[layer_name]
    layer.register_forward_hook(get_activation(layer_name))

    # 前向传播
    with torch.no_grad():
        output = model(input_image)

    # 获取特征图
    feature_maps = activation[layer_name].squeeze(0)

    # 可视化前16个通道
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for idx, ax in enumerate(axes.flat):
        if idx < feature_maps.shape[0]:
            ax.imshow(feature_maps[idx].cpu(), cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Channel {idx}')
    plt.tight_layout()
    plt.show()

# 使用示例
# visualize_feature_maps(model, input_tensor, 'layer1.0.conv1')
```

### 2. 激活值分布可视化

监控每层激活值的分布，检测是否存在梯度消失或爆炸。

```python
import torch.nn as nn

class ActivationMonitor:
    def __init__(self, model):
        self.activations = {}
        self.hooks = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(
                    self.get_activation_hook(name)
                )
                self.hooks.append(hook)

    def get_activation_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = {
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item()
            }
        return hook

    def plot_statistics(self):
        """绘制激活值统计信息"""
        layers = list(self.activations.keys())
        means = [self.activations[l]['mean'] for l in layers]
        stds = [self.activations[l]['std'] for l in layers]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.bar(range(len(layers)), means)
        ax1.set_title('Activation Means')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Mean')

        ax2.bar(range(len(layers)), stds)
        ax2.set_title('Activation Std Devs')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Std')

        plt.tight_layout()
        plt.show()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# 使用示例
# monitor = ActivationMonitor(model)
# output = model(input_tensor)
# monitor.plot_statistics()
# monitor.remove_hooks()
```

### 3. 梯度可视化

检查梯度流动情况，诊断梯度消失或爆炸问题。

```python
def plot_grad_flow(named_parameters):
    """绘制梯度流动图"""
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=max(max_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.legend([
        'max-gradient',
        'mean-gradient',
        'zero-gradient'
    ])
    plt.tight_layout()
    plt.show()

# 使用示例（在backward之后调用）
# loss.backward()
# plot_grad_flow(model.named_parameters())
```

### 4. 注意力图可视化（Attention Maps）

对于使用注意力机制的模型，可视化注意力权重。

```python
def visualize_attention(attention_weights, input_image):
    """可视化注意力图"""
    # attention_weights: [batch, num_heads, seq_len, seq_len]

    # 平均所有注意力头
    attention = attention_weights.mean(dim=1)[0]  # [seq_len, seq_len]

    # 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(attention.cpu().detach().numpy(), cmap='hot')
    plt.colorbar()
    plt.title('Attention Weights')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.show()
```

### 5. 类激活映射（CAM/Grad-CAM）

可视化模型关注图像的哪些区域做出决策。

```python
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        # 前向传播
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1)

        # 反向传播
        self.model.zero_grad()
        output[0, target_class].backward()

        # 计算权重
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # 加权求和
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # 归一化
        cam = cam - cam.min()
        cam = cam / cam.max()

        # 上采样到输入图像大小
        cam = F.interpolate(
            cam,
            size=input_image.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        return cam.squeeze().cpu().numpy()

# 使用示例
# grad_cam = GradCAM(model, model.layer4[-1])
# cam = grad_cam.generate_cam(input_tensor)
# plt.imshow(cam, cmap='jet', alpha=0.5)
```

## 调试技巧

### 1. 检查数据加载

```python
def check_dataloader(dataloader, num_batches=3):
    """检查数据加载器"""
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches:
            break

        print(f"Batch {i}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Images range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels[:10]}")
        print()
```

### 2. 监控训练指标

```python
class TrainingMonitor:
    def __init__(self):
        self.losses = []
        self.accuracies = []
        self.learning_rates = []

    def update(self, loss, accuracy, lr):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.learning_rates.append(lr)

    def plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(self.losses)
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)

        axes[1].plot(self.accuracies)
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True)

        axes[2].plot(self.learning_rates)
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('LR')
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()
```

### 3. 检查模型输出

```python
def check_model_output(model, input_tensor):
    """检查模型输出的合理性"""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    print("Output Statistics:")
    print(f"  Shape: {output.shape}")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    print(f"  Contains NaN: {torch.isnan(output).any().item()}")
    print(f"  Contains Inf: {torch.isinf(output).any().item()}")
```

## 常用工具库

### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

# 记录标量
writer.add_scalar('Loss/train', loss, epoch)

# 记录图像
writer.add_image('predictions', img_grid, epoch)

# 记录模型图
writer.add_graph(model, input_tensor)

# 记录直方图
writer.add_histogram('conv1.weight', model.conv1.weight, epoch)

writer.close()
```

### Weights & Biases (wandb)

```python
import wandb

# 初始化
wandb.init(project="my-project", name="experiment-1")

# 记录指标
wandb.log({
    "loss": loss,
    "accuracy": accuracy,
    "learning_rate": lr
})

# 记录图像
wandb.log({"examples": [wandb.Image(img) for img in images]})

# 保存模型
wandb.save('model.pth')
```

## 调试检查清单

训练模型时，按以下顺序检查：

1. **数据检查**
   - 数据范围是否正确（归一化）
   - 标签是否正确
   - 数据增强是否合理

2. **模型检查**
   - 模型输出shape是否正确
   - 是否有NaN或Inf
   - 初始化是否合理

3. **损失函数检查**
   - 损失值是否在合理范围
   - 损失是否下降
   - 是否有梯度

4. **优化器检查**
   - 学习率是否合适
   - 梯度是否正常
   - 参数是否更新

5. **训练过程检查**
   - 过拟合/欠拟合
   - 训练/验证曲线
   - 收敛速度

## 总结

特征可视化和调试是深度学习开发中不可或缺的技能。通过合理使用这些工具和技巧，可以：

- 更好地理解模型行为
- 快速定位和解决问题
- 优化模型性能
- 提升模型可解释性

建议在项目中建立系统的可视化和监控流程，及时发现和解决问题。

## 参考资源

- [PyTorch Visualization Tutorial](https://pytorch.org/tutorials/)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
