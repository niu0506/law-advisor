"""
配置文件 — 修改 LLM_PROVIDER 切换模型
所有配置项均可通过 .env 文件或环境变量覆盖
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ========== LLM 提供商选择 ==========
    # 可选值: openai | qianwen | deepseek | zhipu | ollama | custom
    LLM_PROVIDER: str = "custom"

    # ========== OpenAI / 标准 OpenAI 兼容接口 ==========
    OPENAI_API_KEY: str = "sk-your-key"
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    # ========== 通义千问（阿里云）==========
    QIANWEN_API_KEY: str = ""
    QIANWEN_MODEL: str = "qwen-max"

    # ========== DeepSeek ==========
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # ========== 智谱 AI（GLM 系列）==========
    ZHIPU_API_KEY: str = ""
    ZHIPU_MODEL: str = "glm-4"

    # ========== Ollama 本地部署（无需 API Key）==========
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = ""

    # ========== 自定义 OpenAI 兼容接口 ==========
    CUSTOM_API_KEY: str = ""
    CUSTOM_BASE_URL: str = ""
    CUSTOM_MODEL: str = ""

    # ========== 嵌入模型 ==========
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"

    # ========== Reranker 重排序模型 ==========
    # 开启后先检索更多候选再精排，显著提升召回质量；设为空字符串可禁用
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    # 重排候选集倍数：实际检索 TOP_K × RERANK_CANDIDATE_MULT 个候选，再取 TOP_K
    RERANK_CANDIDATE_MULT: int = 3
    # 置信度阈值：最高分低于此值时，回答末尾附加"匹配度较低"警告
    CONFIDENCE_THRESHOLD: float = 0.3

    # ========== RAG 检索参数 ==========
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5

    # ========== 多轮对话 ==========
    # 滑动窗口保留的历史轮数（每轮 = 1 次 user + 1 次 assistant）
    CONVERSATION_WINDOW: int = 6

    # ========== 文件路径 ==========
    CHROMA_DB_PATH: str = "./chroma_db"
    LAWS_DIR: str = "./data/laws"
    # SQLite 对话历史持久化路径
    HISTORY_DB_PATH: str = "./history.db"
    # 已处理文件的 MD5 缓存，用于增量入库（跳过未变更文件）
    FILE_HASH_CACHE: str = "./processed_files.json"

    # ========== 服务配置 ==========
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list = ["*"]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
