# Refuse Wrong 的随笔 | Personal Blog

> **"拒绝平庸，记录成长的每一天。"**

这是一个基于 [Jekyll](https://jekyllrb.com/) 构建的静态个人博客站点，部署于 GitHub Pages。本项目在底层逻辑上进行了深度定制，集成了数学公式渲染、自动化专栏归档、站内搜索及评论系统，旨在构建一个功能完善的个人知识库。

## 🛠 技术栈与功能特性 (Tech & Features)

### 核心功能
- **静态页面生成**：基于 **Jekyll** 框架，无需数据库，安全且易于维护。
- **数学公式支持**：内置 **MathJax** 引擎，支持 LaTeX 语法渲染，可完美显示行内（`$...$`）及块级（`$$...$$`）数学公式，适合理工科笔记记录。
- **无限级专栏系统**：定制化 `column` 布局，基于文章的 `categories` 属性自动生成递归的目录结构，实现知识体系的层级化管理。
- **全文检索**：基于 `search.json` 实现的纯前端即时搜索功能，支持对文章标题、正文内容的快速索引与高亮匹配。

### 交互与集成
- **评论系统**：集成 **[Giscus](giscus.app/zh-CN)**，利用 GitHub Discussions API 存储和管理评论，支持 Markdown 语法。可以参考笔记“[搭建流程](/_posts/2026-01-15-first-day.md)”
- **Live2D 组件**：集成了 [Live2D](https://github.com/stevenjoezhang/live2d-widget.git) 看板娘插件，提供基础的网页互动功能。
- **外部 API 集成**：
  - **Hitokoto**：调用一言 API，每日自动更新首页签名。
  - **社交矩阵**：封装了 GitHub、邮件等社交链接入口。

### 工程化
- **CI/CD 自动化**：配置了 GitHub Actions (`deploy.yml`)，推送代码至 `main` 分支时自动触发构建并部署至 `gh-pages`。
- **国内镜像加速**：`Gemfile` 已预配置 Ruby China 镜像源，提升依赖安装速度。

## 📂 目录结构说明

```text
.
├── _config.yml          # 站点核心配置（插件、元数据、第三方服务ID）
├── _layouts/            # 页面逻辑模板
│   ├── default.html     # 全局基础模板（含 Head、Scripts 引用）
│   ├── post.html        # 文章详情页模板（含 MathJax、Giscus 注入逻辑）
│   └── column.html      # 专栏分类页模板（处理递归目录逻辑）
├── _posts/              # 文章源文件 (命名规范: YYYY-MM-DD-Title.md)
├── assets/              # 静态资源（Images, CSS, JS）
├── index.html           # 首页入口文件
├── search.json          # 搜索索引生成模板
└── Gemfile              # Ruby 依赖包管理文件
```

## 🚀 快速开始

### 前置要求

- Ruby (推荐版本 2.7+)
- Bundler
- Git

### 本地运行

1. 克隆仓库：
   ```bash
   git clone https://github.com/refuse-wrong/refuse-wrong.github.io.git
   cd refuse-wrong.github.io
   ```

2. 安装依赖：
   ```bash
   bundle install
   ```

3. 启动本地服务器：
   ```bash
   bundle exec jekyll serve
   ```

4. 打开浏览器访问 `http://localhost:4000` 查看站点。

## ✍️ 写作指南

### 1. 创建文章
   在 `_posts/` 目录下创建新文件，命名格式为 `YYYY-MM-DD-title.md`。
### 2. Front Matter 配置
   文章头部需包含 YAML 配置块：
   ```yaml
   ---
   layout: post
   title: "文章标题"
   date: YYYY-MM-DD
   category: 学习（生活，游戏）
   ---
   ```
### 3. 特殊语法支持
#### 数学公式：

行内公式：$ E=mc^2 $

#### 块级公式：

```代码段
$$
\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$
```
#### 图像插入：
为了保持排版美观，建议使用 HTML 标签包裹图片，以实现居中和添加注释：
```HTML
<div style="text-align: center;">
  <img src="/assets/images/分类名/图片名.png" style="width: 90%; height: auto; border-radius: 4px;">
  <p style="color: #666; font-size: 0.9rem; margin-top: 5px;">图片描述文字</p>
</div>
```

## ⚙️ 部署 (Deployment)

本站点使用 GitHub Pages 自动部署。推送代码到 `main` 分支后，站点会在几分钟内更新。具体流程：
1. 修改代码或添加文章。
2. 提交并 Push 到 main 分支。
3. Actions 会自动运行构建脚本，将生成的静态文件发布到 gh-pages 分支。

## 最近更新

- 新增搜索功能，提升用户体验。
- 添加 jekyll-archives 插件，实现分类自动存档。
- 优化导航栏和社交矩阵布局。
- 增加表格边框

## 贡献

欢迎提交 Issue 或 Pull Request 来改进这个博客！

## 许可证

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 许可证。
