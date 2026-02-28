# ⚖️ AI 法律顾问

基于 RAG（检索增强生成）技术的智能法律问答系统。将本地法律文档向量化入库，结合大语言模型，实现精准、有据可查的法律咨询服务。

---

## ✨ 功能特性

- **多格式文档支持**：PDF、Word、PPT、Excel、TXT、Markdown，自动解析入库
- **法律条文智能分块**：识别"第X条"结构，保留条文完整性，提升检索质量
- **多 LLM 提供商**：支持 OpenAI、通义千问、DeepSeek、智谱 AI、Ollama 本地部署及任意自定义 OpenAI 兼容接口
- **流式回答**：基于 SSE 的打字机效果，实时推送 LLM 生成内容
- **结构化输出**：每次回答均包含法律分析、适用条文、法律结论、实务建议四个模块
- **来源可溯源**：回答同时返回命中的法律条文出处，有据可查
- **文档管理**：支持通过 Web 界面上传文档、查看已加载法律列表、按法律名称删除、重建向量库

---

## 🗂️ 项目结构

```
.
├── main.py              # FastAPI 应用入口，路由定义
├── rag_engine.py        # RAG 核心引擎（检索 + 生成）
├── document_loader.py   # 文档解析与智能分块
├── llm_client.py        # LLM 客户端工厂
├── config.py            # 配置管理（支持 .env 覆盖）
├── models.py            # Pydantic 数据模型
├── index.html           # 前端页面
├── requirements.txt     # Python 依赖
├── _env.example         # 环境变量示例
├── data/
│   └── laws/            # 法律文档目录（启动时自动扫描）
└── chroma_db/           # ChromaDB 向量库（自动创建）
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 首次运行时会自动下载嵌入模型 `BAAI/bge-large-zh-v1.5`（约 1.3GB），请确保网络畅通或提前下载至本地。

### 2. 配置环境变量

复制示例配置文件并按需修改：

```bash
cp _env.example .env
```

编辑 `.env`，选择你的 LLM 提供商（详见下方配置说明）。

### 3. 放入法律文档（可选）

将 PDF、Word 等格式的法律文件放入 `data/laws/` 目录，启动时会自动扫描加载。也可以在服务启动后通过 Web 界面上传。

```bash
mkdir -p data/laws
# 将你的法律文档复制到此目录
```

### 4. 启动服务

```bash
python main.py
```

服务启动后访问 [http://localhost:8000](http://localhost:8000) 即可使用 Web 界面。

---

## ⚙️ LLM 配置说明

在 `.env` 中设置 `LLM_PROVIDER` 切换模型提供商：

### Ollama（本地部署，无需 API Key）

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

需提前安装 [Ollama](https://ollama.com) 并拉取模型：`ollama pull qwen2.5:7b`

### OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://api.openai.com/v1  # 可替换为国内代理
```

### 通义千问（阿里云）

```env
LLM_PROVIDER=qianwen
QIANWEN_API_KEY=sk-xxx
QIANWEN_MODEL=qwen-max
```

### DeepSeek

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-chat
```

### 智谱 AI（GLM 系列）

```env
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=xxx
ZHIPU_MODEL=glm-4
```

### 自定义 OpenAI 兼容接口

```env
LLM_PROVIDER=custom
CUSTOM_API_KEY=sk-xxx
CUSTOM_BASE_URL=https://your-api.example.com/v1
CUSTOM_MODEL=your-model-name
```

---

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 前端页面 |
| `GET` | `/api/status` | 系统状态（引擎、LLM、文档数等） |
| `POST` | `/api/query` | 非流式问答 |
| `POST` | `/api/query/stream` | 流式问答（SSE） |
| `POST` | `/api/upload` | 上传法律文档 |
| `GET` | `/api/laws` | 已加载法律列表 |
| `DELETE` | `/api/laws/{law_name}` | 删除指定法律 |
| `DELETE` | `/api/rebuild` | 后台重建向量库 |
| `GET` | `/api/health` | 健康检查 |

### 问答接口示例

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "劳动合同试用期最长可以约定多久？"}'
```

---

## 🔧 高级配置

以下参数均可在 `.env` 中覆盖：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_MODEL` | `BAAI/bge-large-zh-v1.5` | 向量化模型，中文效果较好 |
| `CHUNK_SIZE` | `500` | 文档分块最大字符数 |
| `CHUNK_OVERLAP` | `50` | 相邻分块重叠字符数 |
| `TOP_K` | `5` | 每次检索返回的最相关片段数 |
| `CHROMA_DB_PATH` | `./chroma_db` | 向量库持久化路径 |
| `LAWS_DIR` | `./data/laws` | 法律文档扫描目录 |
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `8000` | 服务端口 |

---

## 📋 系统要求

- Python 3.9+
- 内存：建议 4GB 以上（嵌入模型加载约需 1.5GB）
- 磁盘：嵌入模型约 1.3GB + 向量库空间

---

## ⚠️ 免责声明

本系统基于 AI 技术，仅供参考，不构成正式法律意见。重要法律事务请咨询持牌律师。
