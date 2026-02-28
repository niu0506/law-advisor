"""
LLM 客户端工厂 — 根据配置动态创建 LangChain LLM 实例

支持的提供商（通过 config.LLM_PROVIDER 切换）：
  openai | qianwen | deepseek | zhipu | ollama | custom
"""
from config import settings

# 标准 OpenAI 兼容提供商配置表，消除重复代码
_OPENAI_COMPAT = {
    "openai":   lambda s: dict(model=s.OPENAI_MODEL,   api_key=s.OPENAI_API_KEY,   base_url=s.OPENAI_BASE_URL),
    "qianwen":  lambda s: dict(model=s.QIANWEN_MODEL,  api_key=s.QIANWEN_API_KEY,  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "deepseek": lambda s: dict(model=s.DEEPSEEK_MODEL, api_key=s.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1"),
    "zhipu":    lambda s: dict(model=s.ZHIPU_MODEL,    api_key=s.ZHIPU_API_KEY,    base_url="https://open.bigmodel.cn/api/paas/v4"),
    "custom":   lambda s: dict(model=s.CUSTOM_MODEL,   api_key=s.CUSTOM_API_KEY,   base_url=s.CUSTOM_BASE_URL),
}

_PROVIDER_NAMES = {
    "openai": "OpenAI", "qianwen": "通义千问", "deepseek": "DeepSeek",
    "zhipu": "智谱 AI", "ollama": "Ollama", "custom": "自定义",
}

_PROVIDER_MODELS = {
    "openai": "OPENAI_MODEL", "qianwen": "QIANWEN_MODEL", "deepseek": "DEEPSEEK_MODEL",
    "zhipu": "ZHIPU_MODEL", "ollama": "OLLAMA_MODEL", "custom": "CUSTOM_MODEL",
}


def get_llm():
    """
    工厂函数：根据 settings.LLM_PROVIDER 返回对应的 LangChain Chat 模型实例

    Returns:
        LangChain BaseChatModel 实例

    Raises:
        ValueError: 当 LLM_PROVIDER 为未知值时抛出
    """
    p = settings.LLM_PROVIDER.lower()

    if p in _OPENAI_COMPAT:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(**_OPENAI_COMPAT[p](settings))

    if p == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL)

    raise ValueError(f"未知 LLM 提供商: {p}，请检查 config.LLM_PROVIDER 配置")


def get_llm_info() -> dict:
    """获取当前 LLM 配置的可读描述信息，用于前端状态展示"""
    p = settings.LLM_PROVIDER.lower()
    model_attr = _PROVIDER_MODELS.get(p, "")
    return {
        "provider": p,
        "name":     _PROVIDER_NAMES.get(p, p),
        "model":    getattr(settings, model_attr, "unknown") if model_attr else "unknown",
    }

