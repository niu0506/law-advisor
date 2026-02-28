"""
LLM 客户端 — 支持 8 种提供商，改 config.LLM_PROVIDER 即可切换
"""
from config import settings


def get_llm():
    p = settings.LLM_PROVIDER.lower()
    from langchain_openai import ChatOpenAI

    if p == "openai":
        return ChatOpenAI(model=settings.OPENAI_MODEL, api_key=settings.OPENAI_API_KEY,
                          base_url=settings.OPENAI_BASE_URL, temperature=0.1, max_tokens=2000)
    if p == "qianwen":
        return ChatOpenAI(model=settings.QIANWEN_MODEL, api_key=settings.QIANWEN_API_KEY,
                          base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                          temperature=0.1, max_tokens=2000)
    if p == "deepseek":
        return ChatOpenAI(model=settings.DEEPSEEK_MODEL, api_key=settings.DEEPSEEK_API_KEY,
                          base_url="https://api.deepseek.com/v1", temperature=0.1, max_tokens=2000)
    if p == "zhipu":
        return ChatOpenAI(model=settings.ZHIPU_MODEL, api_key=settings.ZHIPU_API_KEY,
                          base_url="https://open.bigmodel.cn/api/paas/v4",
                          temperature=0.1, max_tokens=2000)
    if p == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL,
                          temperature=0.1, num_predict=2000)
    if p == "custom":
        return ChatOpenAI(model=settings.CUSTOM_MODEL, api_key=settings.CUSTOM_API_KEY,
                          base_url=settings.CUSTOM_BASE_URL, temperature=0.1, max_tokens=2000)
    raise ValueError(f"未知 LLM 提供商: {p}")


def get_llm_info() -> dict:
    p = settings.LLM_PROVIDER.lower()
    names = {"openai": "OpenAI", "qianwen": "通义千问", "deepseek": "DeepSeek",
             "zhipu": "智谱 AI", "ollama": "Ollama", "custom": "自定义"}
    models = {"openai": settings.OPENAI_MODEL, "qianwen": settings.QIANWEN_MODEL,
              "deepseek": settings.DEEPSEEK_MODEL, "zhipu": settings.ZHIPU_MODEL,
              "ollama": settings.OLLAMA_MODEL, "custom": settings.CUSTOM_MODEL}
    return {"provider": p, "name": names.get(p, p), "model": models.get(p, "unknown")}
