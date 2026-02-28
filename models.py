"""
数据模型定义 — FastAPI 请求/响应的 Pydantic 数据结构

所有模型均继承自 Pydantic BaseModel，自动完成：
  - 请求体解析与字段校验
  - 响应序列化为 JSON
  - OpenAPI 文档自动生成
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """问答查询请求体"""
    # 用户提问内容，最短 2 个字符（防止无效查询），最长 1000 字符（防止超长输入）
    question: str = Field(..., min_length=2, max_length=1000)
    # 是否使用流式返回（True → SSE 逐字输出，False → 一次性返回完整答案）
    stream: bool = False


class SourceDocument(BaseModel):
    """检索到的参考法律条文信息"""
    law_name: str     # 法律名称，如"中华人民共和国合同法"
    article: str      # 条文编号，如"第十二条"
    content: str      # 条文内容摘要（截取前 200 字符）
    source_file: str  # 来源文件名，便于溯源


class QueryResponse(BaseModel):
    """问答查询响应体"""
    answer: str                    # LLM 生成的法律分析答案
    sources: List[SourceDocument]  # 检索到的参考条文列表（去重后）
    question: str                  # 回传用户原始问题（便于前端对照展示）
    doc_count: int = 0             # 本次查询命中的文档片段数量
    # 响应时间戳，默认为当前时间的 ISO 8601 格式字符串
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class UploadResponse(BaseModel):
    """文件上传响应体"""
    success: bool          # 上传并解析是否成功
    file: str              # 上传的文件名
    chunks_added: int      # 本次新增的向量化文本块数量
    total_chunks: int      # 向量库中的文本块总数
    law_names: List[str]   # 当前向量库中所有法律名称列表
    message: str           # 操作结果描述信息（成功/失败原因）


class StatusResponse(BaseModel):
    """系统状态响应体，用于 /api/status 接口"""
    initialized: bool          # RAG 引擎是否已完成初始化
    doc_count: int             # 向量库中的文档片段总数
    law_names: List[str]       # 已加载的法律名称列表
    llm_info: Dict[str, Any]   # 当前 LLM 配置信息（提供商、模型名等）
    embedding_model: str       # 当前使用的嵌入模型名称
    chunk_size: int            # 文档分块大小配置值
    top_k: int                 # 每次检索返回的最大文档数量
