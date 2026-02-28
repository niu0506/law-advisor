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
    OPENAI_API_KEY: str = "sk-your-key"        # OpenAI API 密钥
    OPENAI_MODEL: str = "gpt-4o-mini"          # 使用的模型名称
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"  # API 基础地址（可替换为代理地址）

    # ========== 通义千问（阿里云）==========
    QIANWEN_API_KEY: str = ""                  # 通义千问 API 密钥
    QIANWEN_MODEL: str = "qwen-max"            # 模型名称，可选 qwen-turbo、qwen-plus 等

    # ========== DeepSeek ==========
    DEEPSEEK_API_KEY: str = ""                 # DeepSeek API 密钥
    DEEPSEEK_MODEL: str = "deepseek-chat"      # 模型名称

    # ========== 智谱 AI（GLM 系列）==========
    ZHIPU_API_KEY: str = ""                    # 智谱 AI API 密钥
    ZHIPU_MODEL: str = "glm-4"                 # 模型名称，可选 glm-4-flash 等

    # ========== Ollama 本地部署（无需 API Key）==========
    OLLAMA_BASE_URL: str = "http://localhost:11434"  # Ollama 服务地址
    OLLAMA_MODEL: str = ""             # 本地已拉取的模型名称

    # ========== 自定义 OpenAI 兼容接口 ==========
    # 当 LLM_PROVIDER="custom" 时使用，适用于各类第三方 OpenAI 兼容服务
    CUSTOM_API_KEY: str = ""    # 自定义服务的 API 密钥
    CUSTOM_BASE_URL: str = ""   # 自定义服务地址
    CUSTOM_MODEL: str = ""      # 自定义模型名称

    # ========== 嵌入模型（文本向量化）==========
    # 用于将文档和查询转换为向量，支持本地 HuggingFace 模型
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"  # 中文效果较好的向量模型

    # ========== RAG 检索参数 ==========
    CHUNK_SIZE: int = 500       # 每个文档分块的最大字符数
    CHUNK_OVERLAP: int = 50     # 相邻分块之间的重叠字符数，避免语义截断
    TOP_K: int = 5              # 每次检索返回的最相关分块数量

    # ========== 文件路径 ==========
    CHROMA_DB_PATH: str = "./chroma_db"   # ChromaDB 向量数据库持久化存储路径
    LAWS_DIR: str = "./data/laws"         # 法律文档目录，启动时自动扫描加载

    # ========== 服务配置 ==========
    HOST: str = "0.0.0.0"                # 监听地址，0.0.0.0 表示接受所有网络接口的连接
    PORT: int = 8000                     # 服务端口
    CORS_ORIGINS: list = ["*"]           # 跨域允许的来源，生产环境建议改为具体域名

    class Config:
        env_file = ".env"    # 支持从 .env 文件读取配置
        extra = "ignore"     # 忽略 .env 中未定义的多余字段


# 全局单例配置对象，其他模块通过 from config import settings 引入
settings = Settings()
