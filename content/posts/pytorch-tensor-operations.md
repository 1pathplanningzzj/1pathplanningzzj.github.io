---
title: "PyTorch Tensor 操作完全指南"
date: 2025-01-24
draft: false
tags: ["PyTorch", "深度学习", "Tensor", "教程"]
categories: ["深度学习"]
---

PyTorch 中的 Tensor 是深度学习的核心数据结构。本文将详细介绍各种常用的 Tensor 操作，包括形状变换、维度操作等。

## 1. 基础形状变换操作

### 1.1 view() - 视图变换

`view()` 是最常用的形状变换操作，它返回一个新的 tensor，与原 tensor 共享底层数据。

```python
import torch

# 创建一个 2x3 的 tensor
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(f"原始形状: {x.shape}")  # torch.Size([2, 3])

# 变换为 3x2
y = x.view(3, 2)
print(f"view后: {y.shape}")  # torch.Size([3, 2])
print(y)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])

# 使用 -1 自动推断维度
z = x.view(-1)  # 展平为一维
print(f"展平: {z.shape}")  # torch.Size([6])

# 使用 -1 自动推断某一维度
w = x.view(3, -1)
print(f"自动推断: {w.shape}")  # torch.Size([3, 2])
```

**注意事项：**
- `view()` 要求 tensor 在内存中是连续的（contiguous）
- 如果不连续，需要先调用 `.contiguous()` 或使用 `reshape()`

### 1.2 reshape() - 更灵活的形状变换

`reshape()` 与 `view()` 类似，但更加灵活，可以处理非连续的 tensor。

```python
# 创建一个非连续的 tensor
x = torch.randn(3, 4)
y = x.t()  # 转置后变为非连续

# view() 会报错
# z = y.view(12)  # RuntimeError

# reshape() 可以正常工作
z = y.reshape(12)
print(f"reshape成功: {z.shape}")  # torch.Size([12])

# reshape 会在必要时复制数据
w = x.reshape(2, 6)
print(f"reshape: {w.shape}")  # torch.Size([2, 6])
```

**view() vs reshape()：**
- `view()` 只能用于连续的 tensor，速度更快（不复制数据）
- `reshape()` 可以处理任何 tensor，必要时会复制数据
- 如果确定 tensor 是连续的，优先使用 `view()`

### 1.3 flatten() - 展平操作

`flatten()` 用于将多维 tensor 展平为一维或部分展平。

```python
x = torch.randn(2, 3, 4)
print(f"原始形状: {x.shape}")  # torch.Size([2, 3, 4])

# 完全展平
y = x.flatten()
print(f"完全展平: {y.shape}")  # torch.Size([24])

# 从第1维开始展平
z = x.flatten(start_dim=1)
print(f"部分展平: {z.shape}")  # torch.Size([2, 12])

# 指定展平范围
w = x.flatten(start_dim=0, end_dim=1)
print(f"展平前两维: {w.shape}")  # torch.Size([6, 4])
```

## 2. 维度扩展操作

### 2.1 expand() - 广播扩展

`expand()` 通过广播机制扩展 tensor 的维度，不会分配新的内存。

```python
x = torch.tensor([[1], [2], [3]])
print(f"原始形状: {x.shape}")  # torch.Size([3, 1])

# 扩展第二维
y = x.expand(3, 4)
print(f"expand后: {y.shape}")  # torch.Size([3, 4])
print(y)
# tensor([[1, 1, 1, 1],
#         [2, 2, 2, 2],
#         [3, 3, 3, 3]])

# 使用 -1 保持维度不变
z = x.expand(-1, 5)
print(f"expand: {z.shape}")  # torch.Size([3, 5])

# 扩展多个维度
a = torch.tensor([1, 2, 3])
b = a.unsqueeze(0).unsqueeze(0)  # 变为 [1, 1, 3]
c = b.expand(2, 4, 3)
print(f"多维expand: {c.shape}")  # torch.Size([2, 4, 3])
```

**注意事项：**
- `expand()` 只能扩展大小为 1 的维度
- 不会分配新内存，返回的是原数据的视图
- 不能用于缩小维度

### 2.2 repeat() - 重复扩展

`repeat()` 通过实际复制数据来重复 tensor，会分配新内存。

```python
x = torch.tensor([[1, 2], [3, 4]])
print(f"原始形状: {x.shape}")  # torch.Size([2, 2])

# 在每个维度上重复
y = x.repeat(2, 3)
print(f"repeat后: {y.shape}")  # torch.Size([4, 6])
print(y)
# tensor([[1, 2, 1, 2, 1, 2],
#         [3, 4, 3, 4, 3, 4],
#         [1, 2, 1, 2, 1, 2],
#         [3, 4, 3, 4, 3, 4]])

# 只在某些维度重复
z = x.repeat(1, 2)
print(f"单维repeat: {z.shape}")  # torch.Size([2, 4])

# 增加新维度并重复
w = x.repeat(3, 1, 1)
print(f"增维repeat: {w.shape}")  # torch.Size([3, 2, 2])
```

**expand() vs repeat()：**
- `expand()` 不复制数据，内存高效，但只能扩展大小为1的维度
- `repeat()` 复制数据，消耗内存，但可以任意重复
- 能用 `expand()` 的场景优先使用 `expand()`

### 2.3 unsqueeze() - 增加维度

`unsqueeze()` 在指定位置插入一个大小为 1 的维度。

```python
x = torch.tensor([1, 2, 3, 4])
print(f"原始形状: {x.shape}")  # torch.Size([4])

# 在第0维增加维度
y = x.unsqueeze(0)
print(f"unsqueeze(0): {y.shape}")  # torch.Size([1, 4])

# 在第1维增加维度
z = x.unsqueeze(1)
print(f"unsqueeze(1): {z.shape}")  # torch.Size([4, 1])

# 使用负索引
w = x.unsqueeze(-1)
print(f"unsqueeze(-1): {w.shape}")  # torch.Size([4, 1])

# 连续增加多个维度
a = x.unsqueeze(0).unsqueeze(0)
print(f"多次unsqueeze: {a.shape}")  # torch.Size([1, 1, 4])
```

### 2.4 squeeze() - 移除维度

`squeeze()` 移除所有大小为 1 的维度，��指定维度。

```python
x = torch.randn(1, 3, 1, 4, 1)
print(f"原始形状: {x.shape}")  # torch.Size([1, 3, 1, 4, 1])

# 移除所有大小为1的维度
y = x.squeeze()
print(f"squeeze(): {y.shape}")  # torch.Size([3, 4])

# 移除指定维度
z = x.squeeze(0)
print(f"squeeze(0): {z.shape}")  # torch.Size([3, 1, 4, 1])

w = x.squeeze(2)
print(f"squeeze(2): {w.shape}")  # torch.Size([1, 3, 4, 1])

# 如果指定维度不是1，不会报错，返回原tensor
a = x.squeeze(1)
print(f"squeeze非1维度: {a.shape}")  # torch.Size([1, 3, 1, 4, 1])
```

## 3. 维度重排操作

### 3.1 transpose() - 转置两个维度

`transpose()` 交换两个指定维度的位置。

```python
x = torch.randn(2, 3, 4)
print(f"原始形状: {x.shape}")  # torch.Size([2, 3, 4])

# 交换维度0和维度1
y = x.transpose(0, 1)
print(f"transpose(0,1): {y.shape}")  # torch.Size([3, 2, 4])

# 交换维度1和维度2
z = x.transpose(1, 2)
print(f"transpose(1,2): {z.shape}")  # torch.Size([2, 4, 3])

# 对于2D tensor，等价于矩阵转置
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
matrix_t = matrix.transpose(0, 1)
# 或使用简写
matrix_t2 = matrix.t()
print(f"矩阵转置: {matrix_t.shape}")  # torch.Size([3, 2])
```

**注意：** `transpose()` 返回的 tensor 通常是非连续的。

### 3.2 permute() - 任意维度重排

`permute()` 可以同时重新排列所有维度。

```python
x = torch.randn(2, 3, 4, 5)
print(f"原始形状: {x.shape}")  # torch.Size([2, 3, 4, 5])

# 重新排列所有维度
y = x.permute(3, 1, 0, 2)
print(f"permute后: {y.shape}")  # torch.Size([5, 3, 2, 4])

# 常见用法：图像数据格式转换
# 从 (batch, height, width, channels) 转为 (batch, channels, height, width)
image = torch.randn(32, 224, 224, 3)  # NHWC格式
image_chw = image.permute(0, 3, 1, 2)  # 转为NCHW格式
print(f"图像格式转换: {image_chw.shape}")  # torch.Size([32, 3, 224, 224])

# 序列数据转换
# 从 (seq_len, batch, features) ���为 (batch, seq_len, features)
seq = torch.randn(100, 32, 512)
seq_batch_first = seq.permute(1, 0, 2)
print(f"序列格式转换: {seq_batch_first.shape}")  # torch.Size([32, 100, 512])
```

**transpose() vs permute()：**
- `transpose()` 只能交换两个维度
- `permute()` 可以同时重排所有维度
- 两者都返回非连续的 tensor

### 3.3 contiguous() - 内存连续化

许多操作要求 tensor 在内存中是连续的，`contiguous()` 可以确保这一点。

```python
x = torch.randn(3, 4)
y = x.transpose(0, 1)

print(f"转置后是否连续: {y.is_contiguous()}")  # False

# 使其连续
z = y.contiguous()
print(f"contiguous后: {z.is_contiguous()}")  # True

# 现在可以使用 view()
w = z.view(12)
print(f"view成功: {w.shape}")  # torch.Size([12])

# 如果已经连续，contiguous() 不会复制数据
a = torch.randn(3, 4)
print(f"原本连续: {a.is_contiguous()}")  # True
b = a.contiguous()  # 不会复制，返回自身
print(f"a和b是同一对象: {a is b}")  # True
```

## 4. 拼接与分割操作

### 4.1 cat() - 拼接

`torch.cat()` 在指定维度上拼接多个 tensor。

```python
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

# 在维度0上拼接（垂直拼接）
z1 = torch.cat([x, y], dim=0)
print(f"dim=0拼接: {z1.shape}")  # torch.Size([4, 2])
print(z1)
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])

# 在维度1上拼接（水平拼接）
z2 = torch.cat([x, y], dim=1)
print(f"dim=1拼接: {z2.shape}")  # torch.Size([2, 4])
print(z2)
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])

# 拼接多个tensor
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = torch.randn(2, 3)
d = torch.cat([a, b, c], dim=0)
print(f"多tensor拼接: {d.shape}")  # torch.Size([6, 3])
```

### 4.2 stack() - 堆叠

`torch.stack()` 在新维度上堆叠多个 tensor。

```python
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = torch.tensor([7, 8, 9])

# 在维度0上堆叠
s1 = torch.stack([x, y, z], dim=0)
print(f"dim=0堆叠: {s1.shape}")  # torch.Size([3, 3])
print(s1)
# tensor([[1, 2, 3],
#         [4, 5, 6],
#         [7, 8, 9]])

# 在维度1上堆叠
s2 = torch.stack([x, y, z], dim=1)
print(f"dim=1堆叠: {s2.shape}")  # torch.Size([3, 3])
print(s2)
# tensor([[1, 4, 7],
#         [2, 5, 8],
#         [3, 6, 9]])
```

**cat() vs stack()：**
- `cat()` 在已有维度上拼接，不增加维度
- `stack()` 在新维度上堆叠，会增加一个维度
- `stack()` 要求所有 tensor 形状完全相同

### 4.3 split() 和 chunk() - 分割

`split()` 和 `chunk()` 用于将 tensor 分��成多个部分。

```python
x = torch.randn(10, 3)

# split: 指定每份的大小
parts1 = torch.split(x, 3, dim=0)  # 分成大小为3的部分
print(f"split份数: {len(parts1)}")  # 4份
print(f"各份大小: {[p.shape for p in parts1]}")
# [torch.Size([3, 3]), torch.Size([3, 3]), torch.Size([3, 3]), torch.Size([1, 3])]

# split: 指定每份的具体大小
parts2 = torch.split(x, [2, 3, 5], dim=0)
print(f"自定义split: {[p.shape for p in parts2]}")
# [torch.Size([2, 3]), torch.Size([3, 3]), torch.Size([5, 3])]

# chunk: 指定分成几份
parts3 = torch.chunk(x, 3, dim=0)  # 分成3份
print(f"chunk份数: {len(parts3)}")  # 3份
print(f"各份大小: {[p.shape for p in parts3]}")
# [torch.Size([4, 3]), torch.Size([3, 3]), torch.Size([3, 3])]
```

### 4.4 unbind() - 解绑维度

`unbind()` 移除指定维度并返回该维度上的所有切片。

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print(f"原始形状: {x.shape}")  # torch.Size([2, 3])

# 在维度0上解绑
parts1 = torch.unbind(x, dim=0)
print(f"unbind(0)份数: {len(parts1)}")  # 2
print(f"各份形状: {[p.shape for p in parts1]}")  # [torch.Size([3]), torch.Size([3])]
print(parts1[0])  # tensor([1, 2, 3])

# 在维度1上解绑
parts2 = torch.unbind(x, dim=1)
print(f"unbind(1)份数: {len(parts2)}")  # 3
print(f"各份形状: {[p.shape for p in parts2]}")  # [torch.Size([2]), torch.Size([2]), torch.Size([2])]
```

## 5. 索引与切片操作

### 5.1 基础索引

```python
x = torch.randn(3, 4, 5)

# 单个索引
y = x[0]  # 第一个元素
print(f"x[0]: {y.shape}")  # torch.Size([4, 5])

# 多维索引
z = x[0, 1]
print(f"x[0,1]: {z.shape}")  # torch.Size([5])

# 切片
w = x[:, :2, :]  # 第二维取前2个
print(f"切片: {w.shape}")  # torch.Size([3, 2, 5])

# 步长切片
a = x[::2, :, :]  # 第一维每隔一个取
print(f"步长切片: {a.shape}")  # torch.Size([2, 4, 5])
```

### 5.2 高级索引

```python
x = torch.randn(4, 5)

# 布尔索引
mask = x > 0
positive = x[mask]
print(f"布尔索引: {positive.shape}")  # 一维tensor，包含所有正值

# 整数数组索引
indices = torch.tensor([0, 2, 3])
selected = x[indices]
print(f"整数索引: {selected.shape}")  # torch.Size([3, 5])

# 多维整数索引
row_indices = torch.tensor([0, 1, 2])
col_indices = torch.tensor([1, 3, 4])
elements = x[row_indices, col_indices]
print(f"多维索引: {elements.shape}")  # torch.Size([3])
```

### 5.3 select() 和 index_select()

```python
x = torch.randn(3, 4, 5)

# select: 在指定维度选择单个索引
y = x.select(0, 1)  # 在维度0选择索引1
print(f"select: {y.shape}")  # torch.Size([4, 5])

# index_select: 在指定维度选择多个索引
indices = torch.tensor([0, 2])
z = x.index_select(0, indices)
print(f"index_select: {z.shape}")  # torch.Size([2, 4, 5])

# 在不同维度上选择
w = x.index_select(1, torch.tensor([1, 3]))
print(f"维度1选择: {w.shape}")  # torch.Size([3, 2, 5])
```

## 6. 实用技巧与最佳实践

### 6.1 内存效率

```python
# 好的做法：使用 view 和 expand（不复制数据）
x = torch.randn(1, 3, 1, 1)
y = x.expand(32, 3, 224, 224)  # 内存高效

# 避免：使用 repeat（复制数据）
# z = x.repeat(32, 1, 224, 224)  # 消耗大量内存
```

### 6.2 维度操作链式调用

```python
# 复杂的维度变换可以链式调用
x = torch.randn(32, 3, 224, 224)
y = (x.permute(0, 2, 3, 1)  # NCHW -> NHWC
      .reshape(32, -1, 3)    # 展平空间维度
      .transpose(1, 2))      # 交换维度
print(f"链式操作: {y.shape}")  # torch.Size([32, 3, 50176])
```

### 6.3 常见错误与解决方案

```python
# 错误1: view() 用于非连续tensor
x = torch.randn(3, 4).t()
# y = x.view(12)  # RuntimeError
y = x.contiguous().view(12)  # 正确
# 或使用 reshape
y = x.reshape(12)  # 也正确

# 错误2: expand() 用于非1维度
x = torch.randn(3, 4)
# y = x.expand(6, 8)  # RuntimeError
# 正确做法：先unsqueeze再expand
y = x.unsqueeze(0).expand(2, 3, 4)

# 错误3: 维度不匹配的拼接
x = torch.randn(3, 4)
y = torch.randn(3, 5)
# z = torch.cat([x, y], dim=0)  # RuntimeError
z = torch.cat([x, y], dim=1)  # 正确，在维度1上拼接
```

## 7. 总结对比表

| 操作 | 功能 | 是否复制数据 | 是否改变维度数 |
|------|------|------------|--------------|
| `view()` | 改变形状 | 否 | 否 |
| `reshape()` | 改变形状 | 可能 | 否 |
| `transpose()` | 交换两个维度 | 否 | 否 |
| `permute()` | 重排所有维度 | 否 | 否 |
| `expand()` | 广播扩展 | 否 | 否 |
| `repeat()` | 重复复制 | 是 | 可能 |
| `unsqueeze()` | 增加维度 | 否 | 是（+1） |
| `squeeze()` | 移除维度 | 否 | 是（-n） |
| `cat()` | 拼接 | 是 | 否 |
| `stack()` | 堆叠 | 是 | 是（+1） |
| `flatten()` | 展平 | 否 | 是 |

## 8. 实战示例

### 8.1 批处理图像数据

```python
# 单张图像 (H, W, C) -> 批处理 (N, C, H, W)
image = torch.randn(224, 224, 3)
batch = image.unsqueeze(0).permute(0, 3, 1, 2)
print(f"批处理图像: {batch.shape}")  # torch.Size([1, 3, 224, 224])

# 多张图像拼接
images = [torch.randn(224, 224, 3) for _ in range(8)]
batch = torch.stack(images).permute(0, 3, 1, 2)
print(f"多图像批处理: {batch.shape}")  # torch.Size([8, 3, 224, 224])
```

### 8.2 注意力机制中的维度操作

```python
# Q, K, V 的维度变换
batch_size, seq_len, d_model = 32, 100, 512
num_heads = 8
d_k = d_model // num_heads

Q = torch.randn(batch_size, seq_len, d_model)

# 分割成多头
Q_heads = Q.view(batch_size, seq_len, num_heads, d_k)
Q_heads = Q_heads.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
print(f"多头注意力: {Q_heads.shape}")  # torch.Size([32, 8, 100, 64])

# 合并多头
output = Q_heads.transpose(1, 2).contiguous()
output = output.view(batch_size, seq_len, d_model)
print(f"合并后: {output.shape}")  # torch.Size([32, 100, 512])
```

### 8.3 卷积特征图处理

```python
# 特征图展平用于全连接层
features = torch.randn(32, 256, 7, 7)  # (N, C, H, W)

# 方法1: flatten
flat1 = features.flatten(start_dim=1)
print(f"flatten: {flat1.shape}")  # torch.Size([32, 12544])

# 方法2: view
flat2 = features.view(32, -1)
print(f"view: {flat2.shape}")  # torch.Size([32, 12544])

# 方法3: reshape
flat3 = features.reshape(32, -1)
print(f"reshape: {flat3.shape}")  # torch.Size([32, 12544])
```

---

通过掌握这些 tensor 操作，你可以灵活地处理各种深度学习任务中的数据变换需求。记住：
- 优先使用不复制数据的操作（view, expand, transpose等）
- 注意内存连续性问题
- 理解每个操作对维度的影响
- 在实际应用中选择最合适的操作方式
