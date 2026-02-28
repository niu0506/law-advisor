from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    stream: bool = False

class SourceDocument(BaseModel):
    law_name: str
    article: str
    content: str
    source_file: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    question: str
    doc_count: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class UploadResponse(BaseModel):
    success: bool
    file: str
    chunks_added: int
    total_chunks: int
    law_names: List[str]
    message: str

class StatusResponse(BaseModel):
    initialized: bool
    doc_count: int
    law_names: List[str]
    llm_info: Dict[str, Any]
    embedding_model: str
    chunk_size: int
    top_k: int
