# Colab 按钮使用指南

## 功能说明

这个 shortcode 允许你在 Hugo 文章中添加 Google Colab 按钮，让读者可以直接在 Colab 中运行你的代码示例。

## 使用方法

### 方式 1：链接到 GitHub 仓库中的 Notebook（推荐）

```markdown
{{< colab github="用户名/仓库名/blob/分支名/路径/notebook.ipynb" text="按钮文字" >}}
```

**示例**：
```markdown
{{< colab github="1pathplanningzzj/1pathplanningzzj.github.io/blob/main/static/notebooks/jax-vs-pytorch-examples.ipynb" text="🚀 在 Colab 中运行完整示例" >}}
```

### 方式 2：直接链接到 Colab Notebook

```markdown
{{< colab notebook="notebook/路径" text="按钮文字" >}}
```

### 方式 3：占位符按钮（暂无链接）

```markdown
{{< colab text="即将推出" >}}
```

## 参数说明

| 参数 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `github` | 否 | GitHub 仓库中的 notebook 路径 | - |
| `notebook` | 否 | Colab notebook 的直接路径 | - |
| `text` | 否 | 按钮显示的文字 | "在 Colab 中运行" |

## 工作流程

### 1. 创建 Jupyter Notebook

在 `static/notebooks/` 目录下创建你的 notebook：

```bash
mkdir -p static/notebooks
# 创建或复制你的 .ipynb 文件到这个目录
```

### 2. 提交到 GitHub

```bash
git add static/notebooks/your-notebook.ipynb
git commit -m "Add Jupyter notebook for article"
git push
```

### 3. 在文章中添加按钮

在你的 Markdown 文章中使用 shortcode：

```markdown
---
title: "我的文章"
---

文章内容...

{{< colab github="你的用户名/你的仓库/blob/main/static/notebooks/your-notebook.ipynb" >}}

更多内容...
```

## 示例效果

按钮会显示为：
- 橙色背景（Colab 官方颜色）
- Colab 图标 + 自定义文字
- 鼠标悬停时颜色变深
- 点击后在新标签页打开 Colab

## 注意事项

1. **GitHub 路径格式**：必须包含 `blob/分支名`
   - ✅ 正确：`user/repo/blob/main/path/notebook.ipynb`
   - ❌ 错误：`user/repo/path/notebook.ipynb`

2. **文件必须是 .ipynb 格式**：Colab 只能打开 Jupyter Notebook 文件

3. **公开仓库**：确保你的 GitHub 仓库是公开的，否则读者无法访问

4. **Notebook 依赖**：在 notebook 的第一个 cell 中安装所需的依赖：
   ```python
   !pip install jax jaxlib
   ```

## 高级用法

### 在代码块后添加按钮

```markdown
这是一段 Python 代码：

\`\`\`python
import jax
print("Hello JAX")
\`\`\`

想要运行完整示例？

{{< colab github="..." text="运行此代码" >}}
```

### 多个按钮

你可以在一篇文章中添加多个按钮，链接到不同的 notebooks：

```markdown
## 基础示例
{{< colab github=".../basic-example.ipynb" text="运行基础示例" >}}

## 高级示例
{{< colab github=".../advanced-example.ipynb" text="运行高级示例" >}}
```

## 故障排除

### 按钮不显示
- 检查 shortcode 语法是否正确
- 确保 `layouts/shortcodes/colab.html` 文件存在

### 点击按钮后 Colab 报错
- 检查 GitHub 路径是否正确
- 确保仓库是公开的
- 确保文件是有效的 .ipynb 格式

### 样式问题
- 清除浏览器缓存
- 检查主题是否覆盖了样式

## 相关资源

- [Google Colab 官方文档](https://colab.research.google.com/)
- [Jupyter Notebook 格式](https://nbformat.readthedocs.io/)
- [Hugo Shortcodes 文档](https://gohugo.io/content-management/shortcodes/)
