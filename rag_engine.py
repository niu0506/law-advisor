"""
RAG 引擎核心 — 向量检索 + LLM 生成

完整流程：
  1. initialize()：加载嵌入模型 → 初始化 ChromaDB → 加载/扫描法律文档
  2. query()       ：用户问题 → 向量检索 → 构建 Prompt → LLM 生成答案（一次性）
  3. query_stream()：同上，但以异步生成器逐 Token 流式返回
  4. add_document()：上传新文件 → 解析分块 → 写入向量库
  5. delete_law()  ：按法律名称删除向量库中对应的所有片段
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

# ========== System Prompt 模板 ==========
# 使用 ChatPromptTemplate 以结构化方式注入检索到的法律条文和用户问题
# 固定输出格式（法律分析 / 适用条文 / 建议）便于前端渲染
PROMPT = ChatPromptTemplate.from_template(
"""你是一名专业、严谨的AI法律顾问。请仅依据提供的【参考条文】回答用户问题，不得编造不存在的法律条文。
如果参考条文不足以回答问题，请明确说明“参考条文不足”。
【参考条文】
{context}
【用户问题】
{question}
请按照以下结构进行回答：
【法律分析】
- 结合参考条文逐步分析问题
- 说明法律逻辑及适用条件
【适用条文】
- 列出相关条文编号及核心内容
- 如果有多个条文，请逐条说明
【法律结论】
- 给出明确的法律判断
【实务建议】
- 提供可操作的法律建议
- 如涉及风险或注意事项请说明
要求：
1. 回答必须严谨、专业、逻辑清晰
2. 不得虚构法律条文
3. 若条文信息不足，请明确说明
4. 结构化输出
4. 使用简体中文
"""
)


class RAGEngine:
    """
    RAG（检索增强生成）引擎

    成员变量说明：
      vectorstore  : ChromaDB 向量数据库实例，存储文档向量
      llm          : LangChain LLM 实例，负责生成最终答案
      embeddings   : HuggingFace 嵌入模型，将文本转换为向量
      is_initialized: 标记引擎是否已完成初始化，用于接口层的健康检查
      doc_count    : 向量库中的文档片段总数
      law_names    : 当前向量库中所有法律名称的有序列表
    """

    def __init__(self):
        # 初始时所有组件均为 None，等待 initialize() 调用后赋值
        self.vectorstore: Optional[Chroma] = None
        self.llm = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.is_initialized = False
        self.doc_count = 0
        self.law_names: List[str] = []

    def initialize(self) -> None:
        """
        初始化 RAG 引擎（同步，应在应用启动时调用）

        执行步骤：
          1. 加载 HuggingFace 嵌入模型（首次运行会自动下载模型文件）
          2. 初始化 LLM 客户端
          3. 若向量库已存在则直接加载，否则扫描法律文档目录并建库
          4. 刷新法律名称列表
        """
        logger.info("🚀 初始化 RAG 引擎...")

        # 加载嵌入模型，cpu 设备兼容性更好，normalize_embeddings 保证余弦距离计算正确
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )

        # 根据配置初始化对应的 LLM 客户端
        self.llm = get_llm()

        db = settings.CHROMA_DB_PATH
        if os.path.exists(db) and os.listdir(db):
            # 向量库已存在：直接加载，避免重复解析文档（冷启动加速）
            self.vectorstore = Chroma(
                persist_directory=db,
                embedding_function=self.embeddings,
                collection_name="laws",
            )
            # 使用官方 API 获取文档片段数量
            self.doc_count = len(self.vectorstore.get()['ids'])
            logger.info(f"📂 加载向量库: {self.doc_count} 片段")
        else:
            # 向量库不存在：扫描 LAWS_DIR 目录，解析所有法律文档并建立向量库
            docs = LawDocumentLoader().load_directory(settings.LAWS_DIR)
            if docs:
                # 从文档列表创建向量库并持久化到磁盘
                self.vectorstore = Chroma.from_documents(
                    docs,
                    self.embeddings,
                    persist_directory=db,
                    collection_name="laws",
                )
                self.doc_count = len(docs)
            else:
                # 文档目录为空：创建空向量库，等待用户通过 /api/upload 上传文档
                self.vectorstore = Chroma(
                    persist_directory=db,
                    embedding_function=self.embeddings,
                    collection_name="laws",
                )
                self.doc_count = 0
                logger.info("📭 向量库为空，请上传法律文档")

        self._refresh_names()   # 同步法律名称列表
        self.is_initialized = True
        logger.info("✅ RAG 引擎就绪")

    def _retriever(self):
        """
        创建并返回向量检索器

        使用余弦相似度（similarity）搜索，返回 TOP_K 个最相关片段

        Raises:
            RuntimeError: 向量库未初始化时抛出
        """
        if self.vectorstore is None:
            raise RuntimeError("向量存储未初始化")
        kw = {"k": settings.TOP_K}
        return self.vectorstore.as_retriever(search_type="similarity", search_kwargs=kw)

    @staticmethod
    def _context(docs: List[Document]) -> str:
        """
        将检索到的文档列表格式化为 LLM Prompt 中的上下文文本

        每个片段以"【法律名 条文编号】"为标题，方便 LLM 引用具体条文

        Args:
            docs: 检索到的 Document 列表

        Returns:
            格式化后的上下文字符串
        """
        return "\n\n".join(
            f"【{d.metadata.get('law_name','')} {d.metadata.get('article','')}】\n{d.page_content}"
            for i, d in enumerate(docs, 1)
        )

    async def query(self, question: str) -> Dict[str, Any]:
        """
        非流式问答：检索 + 生成，一次性返回完整答案

        Args:
            question: 用户提问

        Returns:
            包含 answer、sources、question、doc_count 的字典

        Raises:
            RuntimeError: 引擎未初始化时抛出
        """
        if self.vectorstore is None or self.llm is None:
            raise RuntimeError("RAG引擎未初始化")

        # Step 1: 向量检索，找到最相关的法律条文片段
        docs = self._retriever().invoke(question)
        if not docs:
            # 向量库为空或问题与文档相关性极低时的友好提示
            return {
                "answer": "未找到相关法律条文，建议咨询专业律师或上传相关法律文档。",
                "sources": [],
                "question": question,
                "doc_count": 0,
            }

        # Step 2: 构建包含检索结果和用户问题的完整 Prompt
        prompt = PROMPT.format_messages(context=self._context(docs), question=question)

        # Step 3: 异步调用 LLM 生成答案（ainvoke 不阻塞事件循环）
        resp = await self.llm.ainvoke(prompt)

        return {
            "answer":    resp.content,
            "sources":   self._sources(docs),
            "question":  question,
            "doc_count": len(docs),
        }

    async def query_stream(self, question: str) -> AsyncIterator[str]:
        """
        流式问答：以异步生成器逐 Token 返回 LLM 输出

        前端通过 SSE（Server-Sent Events）接收，实现"打字机"效果

        Args:
            question: 用户提问

        Yields:
            LLM 输出的文本片段

        Raises:
            RuntimeError: 引擎未初始化时抛出
        """
        if self.vectorstore is None or self.llm is None:
            raise RuntimeError("RAG引擎未初始化")

        # 向量检索（同非流式，复用同一检索器）
        docs = self._retriever().invoke(question)
        if not docs:
            yield "未找到相关法律条文，建议咨询专业律师或上传相关法律文档。"
            return

        prompt = PROMPT.format_messages(context=self._context(docs), question=question)

        # astream 逐 Token 异步生成，只 yield 非空内容块
        async for chunk in self.llm.astream(prompt):
            if chunk.content:
                yield chunk.content

    @staticmethod
    def _sources(docs: List[Document]) -> List[Dict[str, str]]:
        """
        从检索文档中提取去重后的来源信息列表

        去重依据：法律名 + 条文编号的组合，避免同一条文重复展示

        Args:
            docs: 检索到的 Document 列表

        Returns:
            去重后的来源信息列表，每项包含 law_name、article、content、source_file
        """
        seen, out = set(), []
        for d in docs:
            # 使用"法律名-条文号"作为去重键
            k = f"{d.metadata.get('law_name')}-{d.metadata.get('article')}"
            if k not in seen:
                seen.add(k)
                out.append({
                    "law_name":    d.metadata.get("law_name", ""),
                    "article":     d.metadata.get("article", ""),
                    # 仅截取前 200 字符展示，完整内容已在 LLM 上下文中
                    "content":     d.page_content[:200] + ("..." if len(d.page_content) > 200 else ""),
                    "source_file": d.metadata.get("source", ""),
                })
        return out

    async def add_document(self, file_path: str) -> Dict[str, Any]:
        """
        将新文档添加到向量库

        Args:
            file_path: 待添加文件的本地路径（临时文件，处理完毕后由调用方删除）

        Returns:
            包含 file、chunks_added、total_chunks、law_names 的字典

        Raises:
            RuntimeError: 向量库未初始化
            ValueError:   文档解析失败或内容为空
        """
        if self.vectorstore is None:
            raise RuntimeError("向量存储未初始化")

        # 解析文件并分块
        docs = LawDocumentLoader().load_file(file_path)
        if not docs:
            raise ValueError("文档解析失败或内容为空")

        # 将新片段写入向量库（ChromaDB 会自动生成 UUID 作为文档 ID）
        self.vectorstore.add_documents(docs)
        self.doc_count += len(docs)
        self._refresh_names()  # 更新法律名称列表

        return {
            "file":         Path(file_path).name,
            "chunks_added": len(docs),
            "total_chunks": self.doc_count,
            "law_names":    self.law_names,
        }

    def _refresh_names(self) -> None:
        """
        从向量库元数据中刷新法律名称列表（去重排序）

        在以下操作后调用：initialize、add_document、delete_law
        """
        if self.vectorstore is None:
            self.law_names = []
            return

        try:
            # 仅获取 metadatas，避免拉取大量向量数据
            results = self.vectorstore.get(include=["metadatas"])
            self.law_names = sorted({
                m["law_name"]
                for m in results.get("metadatas", [])
                if m and isinstance(m, dict) and "law_name" in m
            })
        except (KeyError, TypeError, AttributeError) as e:
            # 元数据格式异常时降级处理，不中断服务
            logger.warning(f"刷新法律名称列表失败: {e}")
            self.law_names = []

    def get_status(self) -> Dict[str, Any]:
        """
        返回系统当前状态快照，供 /api/status 接口调用

        Returns:
            包含初始化状态、文档数量、法律列表、LLM 信息等的字典
        """
        return {
            "initialized":     self.is_initialized,
            "doc_count":       self.doc_count,
            "law_names":       self.law_names,
            "llm_info":        get_llm_info(),
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_size":      settings.CHUNK_SIZE,
            "top_k":           settings.TOP_K,
        }

    async def delete_law(self, law_name: str) -> Dict[str, Any]:
        """
        按法律名称删除向量库中所有对应片段

        Args:
            law_name: 要删除的法律名称（需与元数据中的 law_name 字段完全匹配）

        Returns:
            包含 success、message、deleted_count、remaining_docs、law_names 的字典

        Raises:
            ValueError:   法律名称为空，或删除过程中发生异常
            RuntimeError: 向量库未初始化
        """
        if not law_name:
            raise ValueError("法律名称不能为空")

        if self.vectorstore is None:
            raise RuntimeError("向量存储未初始化")

        try:
            # Step 1: 按 law_name 元数据过滤，获取所有匹配文档的 ID
            results = self.vectorstore.get(where={"law_name": law_name})

            if not results or not results.get("ids"):
                # 未找到匹配文档，返回失败状态而非抛出异常（更友好的前端提示）
                return {
                    "success":       False,
                    "message":       f"未找到法律: {law_name}",
                    "deleted_count": 0,
                }

            ids_to_delete = results["ids"]

            # Step 2: 批量删除指定 ID 的文档向量
            self.vectorstore.delete(ids=ids_to_delete)

            # 更新计数器和名称列表
            self.doc_count -= len(ids_to_delete)
            self._refresh_names()

            return {
                "success":        True,
                "message":        f"成功删除法律: {law_name}",
                "deleted_count":  len(ids_to_delete),
                "remaining_docs": self.doc_count,
                "law_names":      self.law_names,
            }
        except (KeyError, ValueError, RuntimeError) as e:
            logger.error(f"删除法律失败: {e}")
            raise ValueError(f"删除法律失败: {str(e)}")


# 全局单例，整个应用共享同一个 RAGEngine 实例（避免重复加载嵌入模型）
rag_engine = RAGEngine()
