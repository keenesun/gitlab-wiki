# DeepWiki

基于 [deepwiki-open](https://github.com/AsyncFuncAI/deepwiki-open) 改造，面向私有代码仓库的内网自部署 Wiki 生成系统。

## 改造点

| | 原项目 | 改造后 |
|---|---|---|
| 向量存储 | FAISS | **ChromaDB**（支持按文件精确删除） |
| 索引更新 | 仅全量重建 | **git diff 驱动的增量更新** |
| 代码分块 | 固定 token | **语义边界切分**（函数/类/CSS 块） |
| LLM 调用 | 依赖 adalflow 多 provider | **LLM_BASE_URL 环境变量，OpenAI 兼容协议即可** |
| 元数据 | 无 | **SQLite**（commit SHA、wiki 页面、文件依赖） |
| 仓库认证 | HTTPS Token | HTTPS + **SSH Deploy Key** |
| 文件过滤 | 内置列表 | 内置 + **.wikignore** + 二进制检测 |
| 依赖 | 依赖 adalflow（~30 万行） | **完全移除 adalflow** |

## 技术栈

| 层 | 选型 |
|---|---|
| 后端框架 | FastAPI |
| LLM | 任意 OpenAI 兼容服务（DeepSeek / Qwen / 硅基流动 / vLLM 等） |
| Embedding | 同上（BAAI/bge-m3 等） |
| 向量存储 | ChromaDB |
| 元数据 | SQLite |
| 代码分块 | Tree-sitter |
| 前端 | Next.js（原项目保留） |
| 部署 | Docker Compose |

## 环境要求

- **Python 3.11+**
- **uv**（Python 包管理器）— `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Node.js 18+** + npm（前端）

## 快速开始

### 1. 配置环境变量

创建 `.env` 文件：

```bash
# LLM — 指向任意 OpenAI 兼容服务
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-your-key
LLM_MODEL=deepseek-chat

# Embedding
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=sk-your-key
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
```

仅此即可。也兼容 Google Gemini、OpenAI、OpenRouter、Ollama、AWS Bedrock、Azure 等 provider（通过 `api/config/generator.json` 切换）。

### 2. 启动

```bash
# 安装 Python 依赖并启动后端（端口 8001）
uv sync
uv run -m api.main

# 另开终端，安装前端依赖并启动（端口 3000）
npm install
npm run dev
```

打开 `http://localhost:3000`，输入仓库地址即可生成 Wiki。

也可直接 `bash run.sh` 一键启动后端。

### 3. Docker

```bash
docker compose up -d
```

数据目录 `~/.deepwiki`（可通过 `DEEPWIKI_DATA_DIR` 修改）：
- `repos/` — 克隆的仓库
- `chromadb/` — 向量数据
- `metadata/deepwiki.sqlite3` — 元数据
- `wikicache/` — Wiki 缓存

## 私有仓库

### HTTPS Token

GitHub / GitLab / Bitbucket 的 Personal Access Token。

### SSH Deploy Key

```bash
ssh-keygen -t ed25519 -f ./deploy_key -N ""
# 将 deploy_key.pub 设为仓库的 Deploy Key
export GIT_SSH_KEY=./deploy_key
```

自动验证密钥权限（600）、扫描 host key、测试连通性。失败立即报错。

## .wikignore

仓库根目录放置 `.wikignore`（类 `.gitignore` 语法）：

```gitignore
*.log
build/
dist/
!important.log
```

内置过滤 `node_modules/`、`.git/`、`*.pyc`、`*.lock` 等 60+ 规则，二进制文件自动检测（MIME + null byte）。

## 增量更新

首次访问全量索引，后续自动 `git pull` 仅处理变更：

```
git diff --name-status
  → D  → ChromaDB 精确删除
  → R  → 删旧 + 索引新
  → A/M → 重新分块 + embedding + upsert
```

成功后推进 `last_indexed_sha`；失败保留错误，下次重试。

## 配置

### LLM Provider

默认用 `LLM_BASE_URL` 直连。也可通过 JSON 切换 — 编辑 `api/config/generator.json`。

### Embedding

编辑 `api/config/embedder.json`。切换模型后若索引指纹不一致，会提示需重建（不会自动覆盖）。

### 访问控制

```bash
DEEPWIKI_AUTH_MODE=true
DEEPWIKI_AUTH_CODE=your-secret
```

### 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `LLM_BASE_URL` | LLM API 地址 | — |
| `LLM_API_KEY` | LLM API 密钥 | — |
| `LLM_MODEL` | LLM 模型名 | `deepseek-chat` |
| `EMBEDDING_BASE_URL` | Embedding API 地址 | — |
| `EMBEDDING_API_KEY` | Embedding API 密钥 | 同 `LLM_API_KEY` |
| `EMBEDDING_MODEL` | Embedding 模型名 | `BAAI/bge-m3` |
| `EMBEDDING_DIM` | 向量维度，设成与模型一致（如 bge-m3=1024）。不设也能跑，设了可在切换模型时提前发现维度冲突 | — |
| `EMBEDDING_NORMALIZE` | 向量是否归一化，设成与模型一致。bge 系列需设为 `true` | `true` |
| `GIT_SSH_KEY` | SSH 私钥路径 | `~/.ssh/id_ed25519` |
| `DEEPWIKI_DATA_DIR` | 数据目录 | `~/.deepwiki` |
| `DEEPWIKI_AUTH_MODE` | 访问控制 | `false` |
| `DEEPWIKI_AUTH_CODE` | 授权码 | — |

## 项目结构

```
pyproject.toml          # uv 项目配置
run.sh                  # 一键启动后端
api/
├── main.py             # 应用入口
├── api.py              # FastAPI 路由
├── config.py           # 配置加载
├── data_pipeline.py    # 索引管道
├── incremental.py      # 增量更新
├── chunker.py          # 代码分块
├── rag.py              # RAG 检索
├── chroma_retriever.py # ChromaDB 检索器
├── ssh_auth.py         # SSH 认证
├── wikignore.py        # 文件过滤
├── types.py            # 核心类型
├── model_types.py      # 模型相关类型
├── model_client.py     # ModelClient 基类
├── ollama_client.py    # Ollama 客户端
├── prompts.py          # 提示词
├── simple_chat.py      # HTTP 聊天
├── websocket_wiki.py   # WebSocket
├── ollama_patch.py     # Ollama 单文档 embedding
├── tools/embedder.py   # Embedder 工厂
├── db/
│   ├── chroma_store.py # ChromaDB 封装
│   └── meta_store.py   # SQLite 元数据
└── config/
    ├── generator.json  # LLM 配置
    ├── embedder.json   # Embedding 配置
    ├── repo.json       # 仓库过滤配置
    └── lang.json       # 语言配置
src/                    # Next.js 前端
```
