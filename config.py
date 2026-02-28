"""
配置文件 — 修改 LLM_PROVIDER 切换模型
"""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM 提供商: openai | qianwen | deepseek | zhipu | ollama | custom
    LLM_PROVIDER: str = "custom"

    # OpenAI / 兼容接口
    OPENAI_API_KEY: str = "sk-your-key"
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"

    # 通义千问
    QIANWEN_API_KEY: str = ""
    QIANWEN_MODEL: str = "qwen-max"

    # DeepSeek
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # 智谱 GLM
    ZHIPU_API_KEY: str = ""
    ZHIPU_MODEL: str = "glm-4"

    # Ollama 本地（默认，无需 Key）
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2:7b"

    # 自定义 OpenAI 兼容接口
    CUSTOM_API_KEY: str = "sk-34f1cbdaa2e04607b03133e72ea4fc0d"
    CUSTOM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    CUSTOM_MODEL: str = "MiniMax-M2.5"

    # 嵌入模型
    EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"

    # RAG 参数
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5

    # 路径
    CHROMA_DB_PATH: str = "./chroma_db"
    LAWS_DIR: str = "./data/laws"

    # 服务
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list = ["*"]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
