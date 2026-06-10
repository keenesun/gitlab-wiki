# DeepWiki

基于 [deepwiki-open](https://github.com/AsyncFuncAI/deepwiki-open) 改造，面向私有代码仓库的内网自部署 Wiki 生成系统。

## 与原项目的区别

| | 原项目 | 改造后 |
|---|---|---|
| 向量存储 | FAISS（不支持按文件删除） | **ChromaDB**（精确删除 + 增量更新基础） |
| 索引更新 | 仅全量重建 | **git diff 驱动的增量更新** |
| 代码分块 | 固定 token 切分 | **按语言语义边界切分**（函数/类/CSS 块） |
| LLM 调用 | Google / OpenAI 等多 provider | **LLM_BASE_URL 环境变量，接任意 OpenAI 兼容服务** |
| 元数据 | 无 | **SQLite**（commit SHA、wiki 页面、文件依赖） |
| 仓库认证 | HTTPS Token | HTTPS Token + **SSH Deploy Key** |
| 文件过滤 | 内置排除列表 | 内置排除 + **.wikignore** + 二进制检测 |

## 技术栈

| 层 | 选型 |
|---|---|
| 后端框架 | FastAPI |
| LLM | LiteLLM / OpenAI 兼容格式（DeepSeek、Qwen、硅基流动等） |
| Embedding | OpenAI 兼容格式（BAAI/bge-m3 等） |
| 向量存储 | ChromaDB |
| 元数据 | SQLite |
| 代码分块 | Tree-sitter（语义边界） |
| 前端 | Next.js（保留原项目） |
| 部署 | Docker Compose |

## 快速开始

### 1. 配置环境变量

```bash
# LLM — 指向任意 OpenAI 兼容服务
export LLM_BASE_URL=https://api.deepseek.com/v1
export LLM_API_KEY=sk-your-key
export LLM_MODEL=deepseek-chat

# Embedding — 指向任意 OpenAI 兼容服务
export EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
export EMBEDDING_API_KEY=sk-your-key
export EMBEDDING_MODEL=BAAI/bge-m3
export EMBEDDING_DIM=1024
```

只需以上环境变量即可使用。也支持 Google Gemini、OpenAI、OpenRouter、Ollama、AWS Bedrock、Azure、DashScope 等多种 provider。

### 2. 启动

```bash
# 安装依赖
pip install poetry && poetry install -C api
npm install

# 启动后端（端口 8001）
uv run -m api.main

# 启动前端（端口 3000）
npm run dev
```

打开 `http://localhost:3000`，输入仓库地址即可生成 Wiki。

### 3. Docker 部署

```bash
docker compose up -d
```

数据持久化在 `~/.deepwiki`：
- `~/.deepwiki/repos/` — 克隆的仓库
- `~/.deepwiki/chromadb/` — 向量数据
- `~/.deepwiki/metadata/` — SQLite 元数据
- `~/.deepwiki/wikicache/` — 生成的 Wiki 缓存

## 私有仓库接入

### HTTPS Token

支持 GitHub / GitLab / Bitbucket 的 Personal Access Token。

### SSH Deploy Key

```bash
# 生成专用密钥对
ssh-keygen -t ed25519 -f ./deploy_key -N ""

# 将 deploy_key.pub 添加为仓库的 Deploy Key

# 指定密钥路径
export GIT_SSH_KEY=./deploy_key
```

系统自动验证密钥权限（600）、扫描 host key、测试连通性，失败立即报错。支持多个 Git 服务器通过 `~/.ssh/config` 配置。

## 文件过滤

在仓库根目录放置 `.wikignore`（类 `.gitignore` 语法）：

```gitignore
# 排除日志
*.log

# 排除构建目录
build/
dist/

# 取消排除重要日志
!important.log
```

内置过滤：`node_modules/`、`vendor/`、`.git/`、`*.pyc`、`*.lock` 等 60+ 条规则，以及二进制文件自动检测（MIME + null byte）。

## 增量更新

首次索引为全量 clone → 分块 → embedding → 写入 ChromaDB。后续访问自动 `git pull`，仅处理变更文件：

```
git diff --name-status
  → DELETED  → ChromaDB 精确删除
  → RENAMED  → 删除旧路径 + 索引新路径
  → ADDED/MODIFIED → 重新分块 + embedding + upsert
```

所有步骤成功后推进 `last_indexed_sha`，失败则保留错误信息，下次重试。

## 配置

### LLM Provider

除 `LLM_BASE_URL` 直连方式外，还可通过 JSON 配置文件切换 provider。编辑 `api/config/generator.json`。

### Embedding

编辑 `api/config/embedder.json` 切换 embedding 模型。切换后若索引指纹（base_url、model、dim、normalize、chunker_version）不一致，会提示需重建索引，不会自动覆盖已有数据。

### 访问控制

```bash
export DEEPWIKI_AUTH_MODE=true
export DEEPWIKI_AUTH_CODE=your-secret
```

### 完整环境变量列表

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_BASE_URL` | LLM API 地址 | — |
| `LLM_API_KEY` | LLM API 密钥 | — |
| `LLM_MODEL` | LLM 模型名 | `deepseek-chat` |
| `EMBEDDING_BASE_URL` | Embedding API 地址 | — |
| `EMBEDDING_API_KEY` | Embedding API 密钥 | 同 `LLM_API_KEY` |
| `EMBEDDING_MODEL` | Embedding 模型名 | `BAAI/bge-m3` |
| `EMBEDDING_DIM` | Embedding 维度 | — |
| `EMBEDDING_NORMALIZE` | 是否归一化 | `true` |
| `GIT_SSH_KEY` | SSH 私钥路径 | `~/.ssh/id_ed25519` |
| `DEEPWIKI_DATA_DIR` | 数据目录 | `~/.deepwiki` |
| `DEEPWIKI_AUTH_MODE` | 访问控制开关 | `false` |
| `DEEPWIKI_AUTH_CODE` | 访问授权码 | — |
| `DEEPWIKI_CONFIG_DIR` | 配置文件目录 | `api/config/` |

## 项目结构

```
api/
├── main.py              # 应用入口
├── api.py               # FastAPI 路由
├── config.py            # 配置加载
├── data_pipeline.py     # 仓库索引管道
├── incremental.py       # 增量更新核心
├── chunker.py           # Tree-sitter 代码分块
├── rag.py               # RAG 检索
├── chroma_retriever.py  # ChromaDB 检索器
├── ssh_auth.py          # SSH 认证
├── wikignore.py         # 文件过滤规则
├── types.py             # 核心数据类型
├── prompts.py           # Deep Research 提示词
├── simple_chat.py       # HTTP 聊天接口
├── websocket_wiki.py    # WebSocket Wiki 接口
├── ollama_patch.py      # Ollama 单文档 embedding
├── tools/embedder.py    # Embedder 工厂
├── db/
│   ├── chroma_store.py  # ChromaDB 封装
│   └── meta_store.py    # SQLite 元数据
└── config/
    ├── generator.json   # LLM provider 配置
    ├── embedder.json    # Embedding 配置
    ├── repo.json        # 仓库过滤配置
    └── lang.json        # 语言配置
src/                     # Next.js 前端（未修改）
```
