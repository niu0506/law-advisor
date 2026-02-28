"""
LLM 客户端工厂 — 根据配置动态创建 LangChain LLM 实例

支持的提供商（通过 config.LLM_PROVIDER 切换）：
  - openai    : OpenAI 官方接口（含 Azure 等兼容代理）
  - qianwen   : 阿里云通义千问（DashScope）
  - deepseek  : DeepSeek API
  - zhipu     : 智谱 AI（GLM 系列）
  - ollama    : 本地 Ollama 部署（无需 API Key）
  - custom    : 任意自定义 OpenAI 兼容接口
"""
from config import settings


def get_llm():
    """
    工厂函数：根据 settings.LLM_PROVIDER 返回对应的 LangChain Chat 模型实例

    所有模型统一使用 LangChain 接口，上层代码无需感知具体提供商差异。
    temperature=0.1 保证回答稳定、减少随机性（法律场景需要准确性优先）
    max_tokens=2000 限制单次回复长度，避免超出上下文窗口

    Returns:
        LangChain BaseChatModel 实例

    Raises:
        ValueError: 当 LLM_PROVIDER 为未知值时抛出
    """
    p = settings.LLM_PROVIDER.lower()
    # 大多数提供商都兼容 OpenAI 接口，统一使用 ChatOpenAI 并指定不同的 base_url
    from langchain_openai import ChatOpenAI

    if p == "openai":
        # OpenAI 官方接口，base_url 可替换为国内代理
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
        )

    if p == "qianwen":
        # 通义千问：使用阿里云 DashScope OpenAI 兼容模式
        return ChatOpenAI(
            model=settings.QIANWEN_MODEL,
            api_key=settings.QIANWEN_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    if p == "deepseek":
        # DeepSeek：性价比较高的国产大模型
        return ChatOpenAI(
            model=settings.DEEPSEEK_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
        )

    if p == "zhipu":
        # 智谱 AI（GLM）：支持 glm-4、glm-4-flash 等
        return ChatOpenAI(
            model=settings.ZHIPU_MODEL,
            api_key=settings.ZHIPU_API_KEY,
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )

    if p == "ollama":
        # Ollama：本地部署方案，无需 API Key，需提前 ollama pull <model>
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

    if p == "custom":
        # 自定义 OpenAI 兼容接口，适合接入各类第三方或私有化部署服务
        return ChatOpenAI(
            model=settings.CUSTOM_MODEL,
            api_key=settings.CUSTOM_API_KEY,
            base_url=settings.CUSTOM_BASE_URL,
        )

    # 未知提供商，给出明确错误提示
    raise ValueError(f"未知 LLM 提供商: {p}，请检查 config.LLM_PROVIDER 配置")


def get_llm_info() -> dict:
    """
    获取当前 LLM 配置的可读描述信息，用于前端状态展示

    Returns:
        包含 provider（标识符）、name（中文名称）、model（模型名）的字典
    """
    p = settings.LLM_PROVIDER.lower()

    # 提供商标识符 → 中文显示名称映射
    names = {
        "openai":   "OpenAI",
        "qianwen":  "通义千问",
        "deepseek": "DeepSeek",
        "zhipu":    "智谱 AI",
        "ollama":   "Ollama",
        "custom":   "自定义",
    }

    # 提供商标识符 → 当前使用的模型名称映射
    models = {
        "openai":   settings.OPENAI_MODEL,
        "qianwen":  settings.QIANWEN_MODEL,
        "deepseek": settings.DEEPSEEK_MODEL,
        "zhipu":    settings.ZHIPU_MODEL,
        "ollama":   settings.OLLAMA_MODEL,
        "custom":   settings.CUSTOM_MODEL,
    }

    return {
        "provider": p,
        "name":     names.get(p, p),           # 未知提供商直接返回原始标识符
        "model":    models.get(p, "unknown"),   # 未知提供商返回 "unknown"
    }
