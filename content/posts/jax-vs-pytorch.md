---
title: "JAX vs PyTorch：深度学习框架对比与选择指南"
date: 2026-01-19
draft: false
tags: ["深度学习", "JAX", "PyTorch", "机器学习", "框架对比"]
categories: ["知识库"]
---

在深度学习领域，JAX 和 PyTorch 是两个备受关注的框架。PyTorch 以其易用性和动态计算图著称，而 JAX 则以函数式编程和高性能计算见长。本文将深入对比两者的特点、优势和适用场景，帮助你做出明智的选择。

{{< colab github="1pathplanningzzj/1pathplanningzzj.github.io/blob/main/static/notebooks/jax-vs-pytorch-examples.ipynb" text="🚀 在 Colab 中运行完整代码示例" >}}

## 核心设计哲学

### PyTorch：易用性优先

PyTorch 的设计理念是"Pythonic"和直观：

- **命令式编程**：代码执行即定义，符合 Python 习惯
- **面向对象**：使用 `nn.Module` 类构建模型
- **动态计算图**：运行时构建图，便于调试
- **生态完善**：丰富的预训练模型和工具库

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
x = torch.randn(32, 10)
output = model(x)  # 直接调用，符合直觉
```

### JAX：函数式 + 高性能

JAX 的设计理念是函数式编程和可组合变换：

- **函数式编程**：纯函数，无副作用
- **可组合变换**：`grad`、`jit`、`vmap` 等变换可自由组合
- **NumPy 兼容**：API 与 NumPy 高度一致
- **XLA 编译**：自动编译优化，性能极致

```python
import jax
import jax.numpy as jnp

def simple_model(params, x):
    W, b = params
    return jnp.dot(x, W) + b

# 自动求梯度
grad_fn = jax.grad(lambda params, x, y: jnp.mean((simple_model(params, x) - y)**2))

# JIT 编译加速
simple_model_jit = jax.jit(simple_model)

# 自动向量化
batched_model = jax.vmap(simple_model, in_axes=(None, 0))
```

## 核心特性对比

| 特性 | PyTorch | JAX |
|------|---------|-----|
| **编程范式** | 面向对象 + 命令式 | 函数式 |
| **计算图** | 动态图（Eager） | 函数变换（可 JIT） |
| **自动微分** | Autograd（反向模式） | Autograd（正向+反向） |
| **编译优化** | TorchScript / torch.compile | XLA（默认） |
| **并行化** | DataParallel / DDP | vmap / pmap |
| **随机数** | 全局状态 | 显式 PRNG key |
| **调试** | 容易（Python debugger） | 较难（JIT 后） |
| **生态** | 非常丰富 | 快速增长 |

## PyTorch 的核心优势

### 1. 易用性和直观性

PyTorch 的 API 设计非常符合 Python 习惯：

```python
# 模型定义直观
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)

# 训练循环清晰
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch.x), batch.y)
        loss.backward()
        optimizer.step()
```

### 2. 丰富的生态系统

- **torchvision**：计算机视觉（ResNet, YOLO 等）
- **torchtext**：自然语言处理
- **torchaudio**：音频处理
- **Hugging Face Transformers**：预训练模型库
- **PyTorch Lightning**：高级训练框架
- **timm**：图像模型库

### 3. 动态计算图

适合需要动态控制流的场景：

```python
def dynamic_model(x, use_dropout=True):
    x = self.layer1(x)

    # 动态控制流
    if use_dropout and self.training:
        x = F.dropout(x, p=0.5)

    # 动态循环
    for i in range(x.size(0)):
        if x[i].sum() > 0:
            x[i] = self.layer2(x[i])

    return x
```

### 4. 工业界广泛采用

- **部署工具**：TorchServe, ONNX
- **移动端**：PyTorch Mobile
- **生产环境**：Meta, Tesla, OpenAI 等大规模使用

## JAX 的核心优势

### 1. 函数变换的可组合性

JAX 的核心是可组合的函数变换：

```python
# grad: 自动微分
grad_fn = jax.grad(loss_fn)

# jit: JIT 编译
fast_fn = jax.jit(loss_fn)

# vmap: 自动向量化
batched_fn = jax.vmap(loss_fn)

# 组合使用
fast_batched_grad = jax.jit(jax.vmap(jax.grad(loss_fn)))
```

### 2. 高性能计算

通过 XLA 编译器实现极致性能：

```python
@jax.jit
def matmul_chain(x, W1, W2, W3):
    # XLA 会自动融合操作，优化内存访问
    return jnp.dot(jnp.dot(jnp.dot(x, W1), W2), W3)

# 性能通常比 PyTorch 快 2-5x
```

**性能优势**：
- **算子融合**：自动合并多个操作
- **内存优化**：减少中间结果存储
- **并行优化**：自动利用硬件并行性

### 3. 自动向量化（vmap）

轻松处理批量数据和集成学习：

```python
# 单样本函数
def predict_single(params, x):
    return model(params, x)

# 自动批处理
predict_batch = jax.vmap(predict_single, in_axes=(None, 0))

# 集成学习：多个模型并行推理
def ensemble_predict(all_params, x):
    # all_params: (num_models, ...)
    # 自动并行化所有模型
    predictions = jax.vmap(predict_single, in_axes=(0, None))(all_params, x)
    return jnp.mean(predictions, axis=0)
```

### 4. 正向模式自动微分

适合高维输出、低维输入的场景：

```python
# 反向模式（PyTorch 默认）：适合 loss (标量) 对 params (高维) 求导
grad_reverse = jax.grad(loss_fn)

# 正向模式：适合 output (高维) 对 input (低维) 求导
jacobian_forward = jax.jacfwd(model_fn)

# 二阶导数
hessian = jax.hessian(loss_fn)
```

### 5. 显式随机数管理

避免全局状态，保证可复现性：

```python
# PyTorch：全局随机状态
torch.manual_seed(42)
x = torch.randn(10)  # 依赖全局状态

# JAX：显式 PRNG key
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
x = jax.random.normal(subkey, (10,))  # 显式传递 key
```

### 6. 多设备并行（pmap）

简洁的数据并行和模型并行：

```python
# 数据并行：自动分配到多个 GPU/TPU
@jax.pmap
def train_step(params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    return loss, grads

# 在 8 个设备上并行训练
losses, grads = train_step(params, batches)  # batches: (8, batch_size, ...)
```

## 关键区别

### 1. 状态管理

**PyTorch**：模型持有状态（参数、缓冲区）

```python
model = MyModel()
model.weight  # 参数存储在模型中
optimizer = torch.optim.Adam(model.parameters())
```

**JAX**：纯函数，参数外部传递

```python
params = init_params()
def model(params, x):  # 参数显式传递
    return jnp.dot(x, params['weight'])

# 需要使用库（如 Flax, Haiku）管理状态
```

### 2. 训练循环

**PyTorch**：命令式，逐步执行

```python
for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

**JAX**：函数式，通常 JIT 编译

```python
@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for batch in dataloader:
    params, opt_state, loss = train_step(params, opt_state, batch)
```

### 3. 调试体验

**PyTorch**：
- ✅ 可以直接使用 `print()`、`pdb` 调试
- ✅ 错误信息清晰，指向具体代码行
- ✅ 动态图，可以随时检查中间结果

**JAX**：
- ❌ JIT 编译后难以调试
- ❌ 错误信息可能指向 XLA 内部
- ✅ 可以用 `jax.disable_jit()` 临时关闭 JIT
- ✅ `jax.debug.print()` 在 JIT 中打印

### 4. 内存管理

**PyTorch**：
- 自动管理 GPU 内存
- 可能出现内存碎片
- `torch.cuda.empty_cache()` 手动清理

**JAX**：
- 预分配 GPU 内存（默认 75%）
- 更高效的内存使用
- 通过 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 控制

## 适用场景

### 选择 PyTorch 的场景

1. **快速原型开发**：需要快速迭代和实验
2. **复杂控制流**：模型包含大量动态逻辑
3. **工业部署**：需要成熟的部署工具链
4. **团队协作**：团队熟悉 PyTorch 生态
5. **预训练模型**：需要使用大量现成模型（Hugging Face）
6. **计算机视觉/NLP**：标准任务，生态完善

**典型应用**：
- 大语言模型训练（GPT, LLaMA）
- 计算机视觉（目标检测、分割）
- 强化学习（动态环境交互）
- 研究原型开发

### 选择 JAX 的场景

1. **高性能计算**：需要极致性能优化
2. **科学计算**：物理模拟、微分方程求解
3. **研究创新**：需要灵活的自动微分（正向、反向、高阶）
4. **TPU 训练**：Google Cloud TPU 优化
5. **集成学习**：需要并行训练多个模型
6. **函数式编程**：偏好纯函数和可组合性

**典型应用**：
- 强化学习（DeepMind 使用 JAX）
- 科学机器学习（物理信息神经网络）
- 贝叶斯推断（概率编程）
- 大规模并行训练（TPU pods）

## 生态系统对比

### PyTorch 生态

**核心库**：
- `torch`：核心框架
- `torchvision`、`torchtext`、`torchaudio`：领域库
- `torch.distributed`：分布式训练

**高级框架**：
- PyTorch Lightning：简化训练流程
- Hugging Face Transformers：预训练模型
- timm：图像模型库
- MMDetection：目标检测工具箱

**部署工具**：
- TorchServe：模型服务
- ONNX：模型转换
- PyTorch Mobile：移动端部署

### JAX 生态

**核心库**：
- `jax`：核心框架
- `jax.numpy`：NumPy 兼容 API
- `optax`：优化器库

**神经网络库**：
- **Flax**：官方推荐，灵活且高性能
- **Haiku**：DeepMind 开发，Sonnet 风格
- **Equinox**：现代化设计，PyTorch 风格

**专用库**：
- **RLax**：强化学习
- **Chex**：测试和调试工具
- **Orbax**：检查点和序列化
- **JAX-MD**：分子动力学模拟

## 性能对比

### 训练速度

典型场景（ResNet-50, ImageNet）：

| 框架 | GPU (V100) | TPU v3 |
|------|-----------|--------|
| PyTorch | 100% | - |
| PyTorch (torch.compile) | 120% | - |
| JAX | 130% | 200% |

**JAX 优势场景**：
- 小批量训练（XLA 优化更明显）
- TPU 训练（原生支持）
- 复杂数学运算（算子融合）

**PyTorch 优势场景**：
- 大批量训练（cuDNN 优化）
- 标准模型（高度优化）
- 动态控制流（无编译开销）

### 内存效率

```python
# JAX：更高效的内存使用
@jax.jit
def efficient_fn(x):
    # XLA 自动优化内存布局
    return jnp.sum(jnp.exp(x) * jnp.log(x))

# PyTorch：需要手动优化
def manual_fn(x):
    # 可能创建多个中间张量
    return torch.sum(torch.exp(x) * torch.log(x))
```

## 代码示例：完整训练流程

### PyTorch 版本

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.layers(x)

# 训练
model = MLP().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
```

### JAX 版本（使用 Flax）

```python
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# 定义模型
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# 初始化
model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# 训练步骤（JIT 编译）
@jax.jit
def train_step(params, opt_state, batch_x, batch_y):
    def loss_fn(params):
        logits = model.apply(params, batch_x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# 训练循环
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        params, opt_state, loss = train_step(params, opt_state, batch_x, batch_y)
```

## 学习曲线

### PyTorch

- **入门**：⭐⭐⭐⭐⭐（非常容易）
- **进阶**：⭐⭐⭐⭐（文档丰富）
- **精通**：⭐⭐⭐（需要理解分布式训练）

**学习路径**：
1. 基础：张量操作、自动微分
2. 模型：`nn.Module`、损失函数、优化器
3. 进阶：自定义层、分布式训练
4. 部署：TorchScript、ONNX

### JAX

- **入门**：⭐⭐⭐（需要理解函数式编程）
- **进阶**：⭐⭐⭐⭐（需要理解 JIT、vmap）
- **精通**：⭐⭐⭐⭐⭐（需要深入理解 XLA）

**学习路径**：
1. 基础：NumPy API、纯函数
2. 变换：`grad`、`jit`、`vmap`
3. 神经网络：选择库（Flax/Haiku）
4. 进阶：`pmap`、自定义梯度、XLA 优化

## 迁移指南

### 从 PyTorch 到 JAX

**主要变化**：
1. **去除类**：用纯函数替代 `nn.Module`
2. **显式参数**：参数不再存储在模型中
3. **JIT 编译**：用 `@jax.jit` 加速
4. **随机数**：使用显式 PRNG key

**迁移步骤**：
```python
# PyTorch
class Model(nn.Module):
    def __init__(self):
        self.weight = nn.Parameter(torch.randn(10, 5))

    def forward(self, x):
        return x @ self.weight

# JAX (Flax)
class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.normal(), (10, 5))
        return x @ weight
```

### 从 JAX 到 PyTorch

**主要变化**：
1. **添加类**：用 `nn.Module` 封装
2. **隐式参数**：参数存储在模型中
3. **去除 JIT**：默认 eager 执行
4. **全局随机数**：使用 `torch.manual_seed()`

## 总结与建议

### 快速决策指南

**选择 PyTorch，如果你**：
- 🚀 需要快速开发和迭代
- 🏢 在工业界部署模型
- 👥 团队已熟悉 PyTorch
- 📚 需要丰富的预训练模型
- 🔧 模型包含复杂控制流

**选择 JAX，如果你**：
- ⚡ 追求极致性能
- 🔬 从事科学计算研究
- 🧮 需要灵活的自动微分
- ☁️ 使用 Google Cloud TPU
- 🎯 偏好函数式编程

### 混合使用

两者并非互斥，可以结合使用：

1. **原型 → 生产**：PyTorch 开发，JAX 优化性能关键部分
2. **研究 → 应用**：JAX 研究新算法，PyTorch 工程化
3. **互操作**：通过 ONNX 或 `jax2torch` 转换模型

### 未来趋势

**PyTorch**：
- `torch.compile`（PyTorch 2.0）缩小性能差距
- 更好的 TPU 支持
- 持续优化易用性

**JAX**：
- 生态系统快速成长
- 更多高级库（Flax、Equinox）
- 工业界采用增加（DeepMind、Google）

## 参考资源

### PyTorch
- [官方文档](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Lightning](https://lightning.ai/)

### JAX
- [官方文档](https://jax.readthedocs.io/)
- [JAX 101 Tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Awesome JAX](https://github.com/n2cholas/awesome-jax)

---

**作者**: zijian
**日期**: 2026-01-19
