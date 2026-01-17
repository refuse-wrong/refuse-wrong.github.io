# Refuse Wrong 的随笔

## 简介

这是一个基于 Jekyll 的个人博客站点，使用 Minima 主题。博客主题是“拒绝平庸，记录成长的每一天。”，旨在分享学习笔记、日常感悟和技术心得。

## 功能特性

- **简洁设计**：使用 Jekyll Minima 主题，提供干净的阅读体验。
- **Markdown 支持**：所有文章使用 Markdown 格式编写。
- **评论系统**：集成 Giscus 评论功能，基于 GitHub Discussions。
- **SEO 优化**：内置 jekyll-seo-tag 插件，提升搜索引擎可见性。
- **RSS 订阅**：支持 jekyll-feed 插件，方便读者订阅更新。

## 安装与运行

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

## 添加新文章

1. 在 `_posts/` 目录下创建新文件，命名格式为 `YYYY-MM-DD-title.md`。
2. 文件开头添加 Front Matter：
   ```yaml
   ---
   layout: post
   title: "文章标题"
   date: YYYY-MM-DD
   category: 学习（生活，游戏）
   ---
   ```
3. 使用 Markdown 编写文章内容。
4. 提交并推送更改，GitHub Pages 会自动更新。

## 部署

本站点使用 GitHub Pages 自动部署。推送代码到 `main` 分支后，站点会在几分钟内更新。

## 配置

主要配置在 `_config.yml` 文件中，包括：
- 站点标题和描述
- 主题设置
- 插件启用

## 贡献

欢迎提交 Issue 或 Pull Request 来改进这个博客！
