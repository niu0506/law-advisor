"""
文档加载器 — MarkItDown 解析 + 法律条文智能分块

主要功能：
  1. 支持 PDF、Word、PPT、Excel、TXT、Markdown 等多种格式
  2. 优先使用 MarkItDown 解析，失败时自动回退到备选解析器
  3. 针对法律条文结构（"第X条"）进行智能分块，保留条文完整性
  4. 增量入库：通过 MD5 哈希缓存跳过未变更文件，大幅加速冷启动
"""
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

logger = logging.getLogger(__name__)

# 支持的文件扩展名集合
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx', '.xlsx', '.txt', '.md'}


def _file_md5(path: str) -> str:
    """计算文件 MD5，用于增量检测"""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_hash_cache() -> Dict[str, str]:
    """加载已处理文件的哈希缓存 {文件绝对路径: md5}"""
    p = Path(settings.FILE_HASH_CACHE)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def _save_hash_cache(cache: Dict[str, str]) -> None:
    """持久化哈希缓存到磁盘"""
    Path(settings.FILE_HASH_CACHE).write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8'
    )


class LawDocumentLoader:
    """法律文档加载器，负责解析文件并拆分为适合向量化的文本块"""

    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n第", "\n##", "\n#", "\n\n", "\n", "。", "；", " ", ""],
        )

    def load_directory(self, directory: str, incremental: bool = True) -> List[Document]:
        """
        递归扫描目录，加载所有支持格式的文件

        Args:
            directory:   文件目录路径，不存在时自动创建
            incremental: True 时启用增量模式，跳过哈希未变更的文件

        Returns:
            新增/变更文件解析后的 Document 列表
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        files = [f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]

        # 增量模式：读取哈希缓存，过滤未变更文件
        hash_cache = _load_hash_cache() if incremental else {}
        new_cache = dict(hash_cache)

        docs = []
        skipped = 0
        for f in files:
            abs_path = str(f.resolve())
            current_md5 = _file_md5(abs_path)

            if incremental and hash_cache.get(abs_path) == current_md5:
                skipped += 1
                continue  # 文件未变更，跳过

            try:
                d = self.load_file(abs_path)
                docs.extend(d)
                new_cache[abs_path] = current_md5  # 成功后才更新缓存
                logger.info(f"✅ {f.name} → {len(d)} 片段")
            except Exception as e:
                logger.error(f"❌ 处理文件 {f.name} 时出错: {str(e)}", exc_info=True)

        if incremental:
            _save_hash_cache(new_cache)

        if skipped:
            logger.info(f"⏭️  跳过 {skipped} 个未变更文件（增量模式）")
        logger.info(f"共加载 {len(docs)} 个新片段")
        return docs

    def load_file(self, file_path: str) -> List[Document]:
        """
        加载单个文件，返回带元数据的 Document 列表

        Args:
            file_path: 文件绝对或相对路径

        Returns:
            切分后的 Document 列表
        """
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
                    "source":   path.name,
                    "law_name": law_name,
                    "article":  self._get_article_tag(c) or f"片段{i+1}",
                }
            )
            for i, c in enumerate(chunks) if c.strip()
        ]

    def _markitdown_convert(self, file_path: str) -> str:
        """使用 MarkItDown 将二进制文档转换为 Markdown 文本，失败时降级到备选解析器"""
        try:
            from markitdown import MarkItDown
            return MarkItDown().convert(file_path).text_content
        except ImportError:
            logger.warning("未安装 markitdown 库，使用备选解析方案")
            return self._fallback(file_path)
        except Exception as e:
            err_msg = str(e)
            if "MissingDependencyException" in err_msg or "dependencies needed" in err_msg:
                logger.warning(
                    f"MarkItDown 缺少依赖（{Path(file_path).suffix}），使用备选解析方案。"
                    f"可运行 pip install 'markitdown[docx,pptx,xlsx]' 修复。"
                )
            else:
                logger.error(f"MarkItDown 解析失败: {e}")
            return self._fallback(file_path)

    @staticmethod
    def _fallback(file_path: str) -> str:
        """备选文档解析器：pypdf / python-docx / 直接读文本"""
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
        """
        法律条文专用分块逻辑

        策略：
          1. 按"第X条"正则分割（保留条文完整性）
          2. 超长条文用通用分割器进一步切分
          3. 不含条文格式时直接用通用分割器
        """
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
        """从文档内容或文件名提取法律名称"""
        lines = content.strip().split('\n')
        if not lines:
            return stem
        first_line = re.sub(r'^[#\s*]+', '', lines[0]).strip()
        if 5 < len(first_line) < 50 and any(k in first_line for k in ('法', '条例', '规定', '办法')):
            return first_line
        return re.sub(r'[-_\s\d]+', '', stem) or stem

    @staticmethod
    def _get_article_tag(text: str) -> str:
        """从分块文本中提取"第X条"标签"""
        m = re.search(r'第([零一二三四五六七八九十百千]+)条', text)
        return f"第{m.group(1)}条" if m else ""
