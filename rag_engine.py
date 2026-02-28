"""
RAG 引擎 — 向量检索 + LLM 生成
"""
import os
import logging
from typing import List, Dict, AsyncIterator, Optional, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings
from document_loader import LawDocumentLoader
from llm_client import get_llm, get_llm_info

logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate.from_template(
"""你是专业的AI法律顾问，基于以下法律条文回答用户问题。

【参考条文】
{context}

【用户问题】
{question}

请按如下格式回答：
**法律分析：** 基于条文对问题进行分析。
**适用条文：** 指出具体条文编号。
**建议：** 给出可操作的法律建议。

""")


class RAGEngine:
    def __init__(self):
        self.vectorstore: Optional[Chroma] = None
        self.llm = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.is_initialized = False
        self.doc_count = 0
        self.law_names: List[str] = []

    def initialize(self) -> None:
        logger.info("🚀 初始化 RAG 引擎...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
        self.llm = get_llm()

        db = settings.CHROMA_DB_PATH
        if os.path.exists(db) and os.listdir(db):
            self.vectorstore = Chroma(persist_directory=db, embedding_function=self.embeddings,
                                      collection_name="laws")
            # 修复1: 使用官方API获取文档数量
            self.doc_count = len(self.vectorstore.get()['ids'])
            logger.info(f"📂 加载向量库: {self.doc_count} 片段")
        else:
            docs = LawDocumentLoader().load_directory(settings.LAWS_DIR)
            if docs:
                self.vectorstore = Chroma.from_documents(
                    docs, self.embeddings, persist_directory=db, collection_name="laws")
                self.doc_count = len(docs)
            else:
                # 空库，等待用户上传
                self.vectorstore = Chroma(persist_directory=db, embedding_function=self.embeddings,
                                          collection_name="laws")
                self.doc_count = 0
                logger.info("📭 向量库为空，请上传法律文档")

        self._refresh_names()
        self.is_initialized = True
        logger.info("✅ RAG 引擎就绪")

    def _retriever(self):
        """获取检索器"""
        if self.vectorstore is None:
            raise RuntimeError("向量存储未初始化")
        kw = {"k": settings.TOP_K}
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs=kw)

    # 修复2: 将不依赖self的方法改为静态方法
    @staticmethod
    def _context(docs: List[Document]) -> str:
        """构建上下文文本"""
        return "\n\n".join(
            f"【{d.metadata.get('law_name','')} {d.metadata.get('article','')}】\n{d.page_content}"
            for i, d in enumerate(docs, 1)
        )

    async def query(self, question: str) -> Dict[str, Any]:
        if self.vectorstore is None or self.llm is None:
            raise RuntimeError("RAG引擎未初始化")

        docs = self._retriever().invoke(question)
        if not docs:
            return {"answer": "未找到相关法律条文，建议咨询专业律师或上传相关法律文档。",
                    "sources": [], "question": question, "doc_count": 0}
        prompt = PROMPT.format_messages(context=self._context(docs), question=question)
        resp = await self.llm.ainvoke(prompt)
        return {"answer": resp.content, "sources": self._sources(docs),
                "question": question, "doc_count": len(docs)}

    async def query_stream(self, question: str) -> AsyncIterator[str]:
        if self.vectorstore is None or self.llm is None:
            raise RuntimeError("RAG引擎未初始化")

        docs = self._retriever().invoke(question)
        if not docs:
            yield "未找到相关法律条文，建议咨询专业律师或上传相关法律文档。"
            return
        prompt = PROMPT.format_messages(context=self._context(docs), question=question)
        async for chunk in self.llm.astream(prompt):
            if chunk.content:
                yield chunk.content

    # 修复3: 将不依赖self的方法改为静态方法
    @staticmethod
    def _sources(docs: List[Document]) -> List[Dict[str, str]]:
        """提取文档来源信息"""
        seen, out = set(), []
        for d in docs:
            k = f"{d.metadata.get('law_name')}-{d.metadata.get('article')}"
            if k not in seen:
                seen.add(k)
                out.append({"law_name": d.metadata.get("law_name",""),
                            "article": d.metadata.get("article",""),
                            "content": d.page_content[:200] + ("..." if len(d.page_content)>200 else ""),
                            "source_file": d.metadata.get("source","")})
        return out

    async def add_document(self, file_path: str) -> Dict[str, Any]:
        if self.vectorstore is None:
            raise RuntimeError("向量存储未初始化")

        docs = LawDocumentLoader().load_file(file_path)
        if not docs:
            raise ValueError("文档解析失败或内容为空")
        self.vectorstore.add_documents(docs)
        self.doc_count += len(docs)
        self._refresh_names()
        return {"file": Path(file_path).name, "chunks_added": len(docs),
                "total_chunks": self.doc_count, "law_names": self.law_names}

    def _refresh_names(self) -> None:
        """刷新法律名称列表"""
        if self.vectorstore is None:
            self.law_names = []
            return

        try:
            # 修复4: 使用官方API获取数据
            results = self.vectorstore.get(include=["metadatas"])
            self.law_names = sorted({
                m["law_name"] for m in results.get("metadatas", [])
                if m and isinstance(m, dict) and "law_name" in m
            })
        except (KeyError, TypeError, AttributeError) as e:
            # 修复5: 捕获具体异常
            logger.warning(f"刷新法律名称列表失败: {e}")
            self.law_names = []

    def get_status(self) -> Dict[str, Any]:
        return {"initialized": self.is_initialized, "doc_count": self.doc_count,
                "law_names": self.law_names, "llm_info": get_llm_info(),
                "embedding_model": settings.EMBEDDING_MODEL, "chunk_size": settings.CHUNK_SIZE,
                "top_k": settings.TOP_K}

    async def delete_law(self, law_name: str) -> Dict[str, Any]:
        if not law_name:
            raise ValueError("法律名称不能为空")

        if self.vectorstore is None:
            raise RuntimeError("向量存储未初始化")

        try:
            # 修复6: 使用官方API删除文档
            # 首先获取要删除的文档ID
            results = self.vectorstore.get(where={"law_name": law_name})

            if not results or not results.get("ids"):
                return {"success": False, "message": f"未找到法律: {law_name}", "deleted_count": 0}

            ids_to_delete = results["ids"]

            # 删除文档
            self.vectorstore.delete(ids=ids_to_delete)

            self.doc_count -= len(ids_to_delete)
            self._refresh_names()

            return {
                "success": True,
                "message": f"成功删除法律: {law_name}",
                "deleted_count": len(ids_to_delete),
                "remaining_docs": self.doc_count,
                "law_names": self.law_names
            }
        except (KeyError, ValueError, RuntimeError) as e:
            # 修复7: 捕获具体异常
            logger.error(f"删除法律失败: {e}")
            raise ValueError(f"删除法律失败: {str(e)}")


rag_engine = RAGEngine()