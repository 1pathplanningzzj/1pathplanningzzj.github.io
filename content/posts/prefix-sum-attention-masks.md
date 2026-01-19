---
title: "前缀和注意力掩码：用数学优雅实现混合注意力机制"
date: 2026-01-19
draft: false
math: true
tags: ["深度学习", "Transformer", "注意力机制", "多模态"]
---

## 核心思想

在多模态模型（如图文混合、视觉-语言-动作模型）中，不同部分的序列需要不同的注意力模式：
- **图像 token / 前缀**：需要双向注意力（每个 token 可以看到所有其他 token）
- **文本生成 / 动作预测**：需要单向因果注意力（只能看到之前的 token）

`make_att_2d_masks` 函数用一种**极其简洁的数学方式**（前缀和比较），在同一个序列中实现了这种复杂的混合注意力机制。

## 代码实现

```python
def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks
```

## 数学原理：前缀和比较

### 1. 核心机制

函数的核心是这一行：
```python
att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
```

这行代码的含义是：**token $i$ 可以注意到 token $j$，当且仅当 $\text{cumsum}[j] \leq \text{cumsum}[i]$**

其中 $\mathrm{cumsum}[i] = \sum_{k=0}^{i} \mathrm{att\_masks}[k]$ 是前缀和。

### 2. 为什么这样设计？

通过控制 `att_masks` 中的 0 和 1，我们可以控制前缀和的增长：
- **att_masks[i] = 0**：cumsum 不增长，token $i$ 与前一个 token 共享相同的"注意力级别"
- **att_masks[i] = 1**：cumsum 增长，token $i$ 进入新的"注意力级别"

由于比较规则是 $\text{cumsum}[j] \leq \text{cumsum}[i]$，这意味着：
- 同一级别的 token 可以互相注意（双向注意力）
- 只能注意到级别 $\leq$ 自己的 token（因果性）

## 三种典型注意力模式

### 模式 1：纯因果注意力（Pure Causal Attention）

```python
att_masks = [1, 1, 1, 1, 1, 1]
cumsum    = [1, 2, 3, 4, 5, 6]
```

注意力矩阵（✓ 表示可以注意）：
```
       j: 0  1  2  3  4  5
i: 0   [✓  ✗  ✗  ✗  ✗  ✗]
   1   [✓  ✓  ✗  ✗  ✗  ✗]
   2   [✓  ✓  ✓  ✗  ✗  ✗]
   3   [✓  ✓  ✓  ✓  ✗  ✗]
   4   [✓  ✓  ✓  ✓  ✓  ✗]
   5   [✓  ✓  ✓  ✓  ✓  ✓]
```

**解释**：每个 token 的 cumsum 都不同，所以只能注意到 cumsum 更小的 token（即之前的 token）。

### 模式 2：前缀-LM 注意力（Prefix-LM Attention）

```python
att_masks = [0, 0, 0, 1, 1, 1]
cumsum    = [0, 0, 0, 1, 2, 3]
```

注意力矩阵：
```
       j: 0  1  2  3  4  5
i: 0   [✓  ✓  ✓  ✗  ✗  ✗]  <- 前缀部分：双向注意力
   1   [✓  ✓  ✓  ✗  ✗  ✗]
   2   [✓  ✓  ✓  ✗  ✗  ✗]
   3   [✓  ✓  ✓  ✓  ✗  ✗]  <- 生成部分：因果注意力
   4   [✓  ✓  ✓  ✓  ✓  ✗]
   5   [✓  ✓  ✓  ✓  ✓  ✓]
```

**解释**：
- Token 0-2 的 cumsum 都是 0，所以它们可以互相注意（双向）
- Token 3-5 的 cumsum 递增，所以它们只能注意到之前的 token（因果）
- 所有生成部分的 token 都可以注意到前缀部分（因为前缀的 cumsum = 0 最小）

### 模式 3：块状因果注意力（Block-wise Causal Attention）

```python
att_masks = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]
cumsum    = [1, 1, 2, 2, 3, 3, 3, 4, 4, 4]
```

注意力矩阵：
```
       j: 0  1  2  3  4  5  6  7  8  9
i: 0   [✓  ✓  ✗  ✗  ✗  ✗  ✗  ✗  ✗  ✗]  <- Block 1
   1   [✓  ✓  ✗  ✗  ✗  ✗  ✗  ✗  ✗  ✗]
   2   [✓  ✓  ✓  ✓  ✗  ✗  ✗  ✗  ✗  ✗]  <- Block 2
   3   [✓  ✓  ✓  ✓  ✗  ✗  ✗  ✗  ✗  ✗]
   4   [✓  ✓  ✓  ✓  ✓  ✓  ✓  ✗  ✗  ✗]  <- Block 3
   5   [✓  ✓  ✓  ✓  ✓  ✓  ✓  ✗  ✗  ✗]
   6   [✓  ✓  ✓  ✓  ✓  ✓  ✓  ✗  ✗  ✗]
   7   [✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓]  <- Block 4
   8   [✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓]
   9   [✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓]
```

**解释**：
- 每个块内的 token 共享相同的 cumsum，所以可以互相注意（块内双向）
- 每个块可以注意到所有之前的块（块间因果）

## Padding 掩码的作用

```python
pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
return att_2d_masks & pad_2d_masks
```

这两行代码确保：
- 只有有效的 token（非 padding）才能参与注意力计算
- Padding token 既不能被注意，也不能注意其他 token

`pad_2d_masks[i, j] = True` 当且仅当 token $i$ 和 token $j$ 都是有效 token。

## 为什么这个设计如此优雅？

### 1. 数学简洁性

用一行前缀和比较，替代了复杂的条件判断和循环：
```python
# 传统方法可能需要：
for i in range(N):
    for j in range(N):
        if is_prefix(j) or (is_causal(j) and j <= i):
            mask[i, j] = True

# 前缀和方法只需要：
att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
```

### 2. 统一的抽象

通过一个简单的 `att_masks` 向量，可以表达任意复杂的混合注意力模式：
- 不需要为每种模式写单独的代码
- 易于扩展到新的注意力模式
- 便于动态调整（例如根据输入长度）

### 3. 高效的并行计算

- 前缀和：$O(N)$ 时间复杂度
- 广播比较：完全并行化，GPU 友好
- 无需循环或条件分支

### 4. 灵活性

可以轻松实现各种复杂场景：
- **多模态模型**：图像（双向）+ 文本（因果）
- **具身智能**：观察（双向）+ 动作（因果）
- **代码生成**：上下文（双向）+ 生成（因果）
- **分层注意力**：不同粒度的块状注意力

## 应用场景

### 1. 视觉-语言模型（VLM）

```python
# 图像 patch: 256 个 token（双向）
# 文本 token: 128 个 token（因果）
att_masks = [0] * 256 + [1] * 128
```

图像部分可以互相注意，文本部分只能看到之前的文本和所有图像。

### 2. 具身智能（Embodied AI）

```python
# 观察（图像 + 状态）: 双向
# 动作序列: 因果
att_masks = [0] * obs_len + [1] * action_len
```

模型可以充分理解当前观察，然后自回归生成动作序列。

### 3. 代码补全

```python
# 已有代码上下文: 双向
# 待生成代码: 因果
att_masks = [0] * context_len + [1] * generation_len
```

模型可以双向理解上下文，然后单向生成新代码。

## 总结

`make_att_2d_masks` 函数展示了如何用**简洁的数学抽象**解决**复杂的工程问题**：

1. **前缀和**：将序列转化为"注意力级别"
2. **广播比较**：高效生成 2D 注意力矩阵
3. **统一接口**：一个函数支持所有混合注意力模式

这种设计哲学在深度学习系统设计中非常重要：**用数学的优雅性，换取工程的简洁性**。
