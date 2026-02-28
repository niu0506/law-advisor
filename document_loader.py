"""
文档加载器 — MarkItDown 解析 + 法律条文智能分块
"""
import re
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

logger = logging.getLogger(__name__)

# 修正：常量名使用大写蛇形命名法 (UPPER_SNAKE_CASE)
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx', '.xlsx', '.txt', '.md'}


class LawDocumentLoader:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n第", "\n##", "\n#", "\n\n", "\n", "。", "；", " ", ""],
        )

    def load_directory(self, directory: str) -> List[Document]:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        # 使用修正后的常量名
        files = [f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        docs = []
        for f in files:
            try:
                d = self.load_file(str(f))
                docs.extend(d)
                logger.info(f"✅ {f.name} → {len(d)} 片段")
            except Exception as e:
                # 细化异常捕获，避免吞掉键盘中断等系统错误
                logger.error(f"❌ 处理文件 {f.name} 时出错: {str(e)}", exc_info=True)

        logger.info(f"共加载 {len(docs)} 个片段")
        return docs

    def load_file(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in {'.pdf', '.docx', '.doc', '.pptx', '.xlsx'}:
            text = self._markitdown_convert(file_path)
        else:
            text = path.read_text(encoding='utf-8', errors='ignore')

        if not text.strip():
            return []

        law_name = self._get_law_name(path.stem, text)
        chunks = self._split_logic(text)

        return [
            Document(
                page_content=c,
                metadata={
                    "source": path.name,
                    "law_name": law_name,
                    "article": self._get_article_tag(c) or f"片段{i+1}",
                }
            )
            for i, c in enumerate(chunks) if c.strip()
        ]

    def _markitdown_convert(self, file_path: str) -> str:
        """尝试使用 MarkItDown 转换，失败则回退"""
        try:
            from markitdown import MarkItDown
            return MarkItDown().convert(file_path).text_content
        except ImportError:
            logger.warning("未安装 markitdown 库，使用备选解析方案")
            return self._fallback(file_path)
        except Exception as e:
            err_msg = str(e)
            # MissingDependencyException 说明 markitdown 缺少可选依赖，直接走备选方案
            # 提示用户安装正确的 extras: pip install "markitdown[docx,pptx,xlsx]"
            if "MissingDependencyException" in err_msg or "dependencies needed" in err_msg:
                logger.warning(
                    f"MarkItDown 缺少依赖（{Path(file_path).suffix}），使用备选解析方案。"
                    f"可运行 pip install 'markitdown[docx,pptx,xlsx]' 修复此问题。"
                )
            else:
                logger.error(f"MarkItDown 解析失败: {e}")
            return self._fallback(file_path)

    @staticmethod
    def _fallback(file_path: str) -> str:
        """修正：改为静态方法，不依赖 self"""
        ext = Path(file_path).suffix.lower()
        try:
            if ext == '.pdf':
                from pypdf import PdfReader
                return "\n".join(p.extract_text() or "" for p in PdfReader(file_path).pages)
            if ext in {'.docx', '.doc'}:
                from docx import Document as DocxDoc
                return "\n".join(p.text for p in DocxDoc(file_path).paragraphs)
        except Exception as e:
            logger.warning(f"备选解析器也失败了: {e}")

        return Path(file_path).read_text(encoding='utf-8', errors='ignore')

    def _split_logic(self, text: str) -> List[str]:
        """法律条文拆分核心逻辑"""
        pat = r'(第[零一二三四五六七八九十百千]+条)'
        parts = re.split(pat, text)

        if len(parts) > 3:
            chunks, cur = [], ""
            for p in parts:
                if re.match(pat, p):
                    if cur.strip() and len(cur) > 20:
                        chunks.append(cur.strip())
                    cur = p
                else:
                    cur += p

                if len(cur) > settings.CHUNK_SIZE * 1.5:
                    chunks.extend(self.splitter.split_text(cur))
                    cur = ""

            if cur.strip():
                chunks.append(cur.strip())
            return [c for c in chunks if len(c.strip()) > 20]

        return self.splitter.split_text(text)

    @staticmethod
    def _get_law_name(stem: str, content: str) -> str:
        """修正：改为静态方法。从标题或文件名提取法律名称"""
        lines = content.strip().split('\n')
        if not lines:
            return stem

        first_line = re.sub(r'^[#\s*]+', '', lines[0]).strip()
        if 5 < len(first_line) < 50 and any(k in first_line for k in ('法', '条例', '规定', '办法')):
            return first_line
        return re.sub(r'[-_\s\d]+', '', stem) or stem

    @staticmethod
    def _get_article_tag(text: str) -> str:
        """修正：改为静态方法。提取‘第X条’标签"""
        m = re.search(r'第([零一二三四五六七八九十百千]+)条', text)
        return f"第{m.group(1)}条" if m else ""