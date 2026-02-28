"""
数据模型定义 — FastAPI 请求/响应的 Pydantic 数据结构
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class ChatMessage(BaseModel):
    """单条对话消息（用于多轮对话上下文传递）"""
    role: str     # "user" 或 "assistant"
    content: str  # 消息内容


class QueryRequest(BaseModel):
    """问答查询请求体"""
    question: str = Field(..., min_length=2, max_length=1000)
    stream: bool = False
    # 多轮对话：客户端传入最近几轮历史，引擎会将其拼入 Prompt
    history: List[ChatMessage] = Field(default_factory=list)
    # 会话 ID：用于关联历史记录，不传则每次生成新 ID
    session_id: Optional[str] = None


class SourceDocument(BaseModel):
    """检索到的参考法律条文信息"""
    law_name: str
    article: str
    content: str      # 截取前 200 字符
    source_file: str
    score: float = 0.0  # 重排序置信度分数（0~1），越高越相关


class QueryResponse(BaseModel):
    """问答查询响应体"""
    answer: str
    sources: List[SourceDocument]
    question: str
    doc_count: int = 0
    # 置信度：最高来源文档的重排序分数；未启用 Reranker 时为 -1
    confidence: float = -1.0
    # 低置信度时附加的警告信息
    confidence_warning: str = ""
    session_id: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class UploadResponse(BaseModel):
    """文件上传响应体"""
    success: bool
    file: str
    chunks_added: int
    total_chunks: int
    law_names: List[str]
    message: str


class StatusResponse(BaseModel):
    """系统状态响应体"""
    initialized: bool
    doc_count: int
    law_names: List[str]
    llm_info: Dict[str, Any]
    embedding_model: str
    reranker_model: str         # 当前使用的重排序模型名称，空字符串表示未启用
    reranker_enabled: bool      # 重排序功能是否已启用
    chunk_size: int
    top_k: int


# ── 历史记录相关 ──

class HistoryRecord(BaseModel):
    """单条历史问答记录"""
    id: int
    session_id: str
    question: str
    answer: str
    sources: List[SourceDocument]
    confidence: float
    timestamp: str


class HistoryListResponse(BaseModel):
    """历史记录列表响应"""
    records: List[HistoryRecord]
    total: int
    page: int
    page_size: int
