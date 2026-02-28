"""
文档加载器 — MarkItDown 解析 + 法律条文智能分块

主要功能：
  1. 支持 PDF、Word、PPT、Excel、TXT、Markdown 等多种格式
  2. 优先使用 MarkItDown 解析，失败时自动回退到备选解析器
  3. 针对法律条文结构（"第X条"）进行智能分块，保留条文完整性
"""
import re
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

logger = logging.getLogger(__name__)

# 支持的文件扩展名集合（常量使用大写蛇形命名）
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx', '.xlsx', '.txt', '.md'}


class LawDocumentLoader:
    """法律文档加载器，负责解析文件并拆分为适合向量化的文本块"""

    def __init__(self):
        # 初始化通用文本分割器，作为法律条文分块的兜底方案
        # separators 按优先级尝试：章节标题 > 段落 > 句子 > 词 > 字符
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n第", "\n##", "\n#", "\n\n", "\n", "。", "；", " ", ""],
        )

    def load_directory(self, directory: str) -> List[Document]:
        """
        递归扫描目录，加载所有支持格式的文件

        Args:
            directory: 文件目录路径，不存在时自动创建

        Returns:
            所有文件解析后的 Document 列表
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)  # 目录不存在时自动创建，避免报错

        # 递归查找目录下所有受支持格式的文件
        files = [f for f in path.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        docs = []
        for f in files:
            try:
                d = self.load_file(str(f))
                docs.extend(d)
                logger.info(f"✅ {f.name} → {len(d)} 片段")
            except Exception as e:
                # 仅捕获普通异常，避免吞掉 KeyboardInterrupt 等系统信号
                logger.error(f"❌ 处理文件 {f.name} 时出错: {str(e)}", exc_info=True)

        logger.info(f"共加载 {len(docs)} 个片段")
        return docs

    def load_file(self, file_path: str) -> List[Document]:
        """
        加载单个文件，返回带元数据的 Document 列表

        Args:
            file_path: 文件绝对或相对路径

        Returns:
            切分后的 Document 列表，每个 Document 包含文本内容和来源元数据
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        # 根据文件类型选择解析方式
        if ext in {'.pdf', '.docx', '.doc', '.pptx', '.xlsx'}:
            # 二进制格式使用 MarkItDown 解析（支持富文本结构提取）
            text = self._markitdown_convert(file_path)
        else:
            # 纯文本格式直接读取，errors='ignore' 防止编码问题导致崩溃
            text = path.read_text(encoding='utf-8', errors='ignore')

        # 空文档直接跳过，避免产生无意义的向量
        if not text.strip():
            return []

        # 从文件名或文档首行提取法律名称（用于元数据标注）
        law_name = self._get_law_name(path.stem, text)
        # 对文本进行智能分块（优先按"第X条"拆分）
        chunks = self._split_logic(text)

        # 为每个分块构建 Document 对象，附加元数据便于后续检索和引用
        return [
            Document(
                page_content=c,
                metadata={
                    "source": path.name,                                    # 来源文件名
                    "law_name": law_name,                                   # 法律/法规名称
                    "article": self._get_article_tag(c) or f"片段{i+1}",  # 条文编号或顺序编号
                }
            )
            for i, c in enumerate(chunks) if c.strip()  # 过滤空白分块
        ]

    def _markitdown_convert(self, file_path: str) -> str:
        """
        使用 MarkItDown 将二进制文档转换为 Markdown 文本

        优先级：MarkItDown → 备选解析器（pypdf / python-docx）
        MarkItDown 能更好地保留文档结构（表格、标题层级等）
        """
        try:
            from markitdown import MarkItDown
            return MarkItDown().convert(file_path).text_content
        except ImportError:
            # markitdown 库未安装，降级到备选方案
            logger.warning("未安装 markitdown 库，使用备选解析方案")
            return self._fallback(file_path)
        except Exception as e:
            err_msg = str(e)
            # MissingDependencyException：markitdown 安装了但缺少对应格式的可选依赖
            # 解决方案：pip install "markitdown[docx,pptx,xlsx]"
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
        """
        备选文档解析器，当 MarkItDown 不可用时使用

        支持：
          - PDF → pypdf（逐页提取文字）
          - DOCX/DOC → python-docx（逐段落提取）
          - 其他格式 → 直接读取文本内容
        """
        ext = Path(file_path).suffix.lower()
        try:
            if ext == '.pdf':
                from pypdf import PdfReader
                # 逐页提取文本并拼接，空页返回空字符串避免 None 错误
                return "\n".join(p.extract_text() or "" for p in PdfReader(file_path).pages)
            if ext in {'.docx', '.doc'}:
                from docx import Document as DocxDoc
                # 逐段落提取文本
                return "\n".join(p.text for p in DocxDoc(file_path).paragraphs)
        except Exception as e:
            logger.warning(f"备选解析器也失败了: {e}")

        # 最终兜底：直接按文本读取（适用于 txt/md 等）
        return Path(file_path).read_text(encoding='utf-8', errors='ignore')

    def _split_logic(self, text: str) -> List[str]:
        """
        法律条文专用分块逻辑

        策略：
          1. 先尝试按"第X条"正则分割（保留条文完整性）
          2. 若超过 1.5 倍 CHUNK_SIZE，则对超长条文再次用通用分割器切分
          3. 若文档不含条文格式，直接使用通用分割器
        """
        # 匹配"第X条"格式，X 为中文数字
        pat = r'(第[零一二三四五六七八九十百千]+条)'
        parts = re.split(pat, text)

        # 找到 3 个以上部分说明文本包含多个法条，启用条文分块策略
        if len(parts) > 3:
            chunks, cur = [], ""
            for p in parts:
                if re.match(pat, p):
                    # 遇到新条目：保存上一条（若有内容且长度足够）
                    if cur.strip() and len(cur) > 20:
                        chunks.append(cur.strip())
                    cur = p  # 以条目编号开始新块
                else:
                    cur += p  # 条文内容追加到当前块

                # 当前块超过阈值，用通用分割器进一步切分，防止单块过大
                if len(cur) > settings.CHUNK_SIZE * 1.5:
                    chunks.extend(self.splitter.split_text(cur))
                    cur = ""

            # 处理最后一个未提交的块
            if cur.strip():
                chunks.append(cur.strip())

            # 过滤掉过短的无意义片段
            return [c for c in chunks if len(c.strip()) > 20]

        # 无法律条文格式，直接使用通用分割器
        return self.splitter.split_text(text)

    @staticmethod
    def _get_law_name(stem: str, content: str) -> str:
        """
        从文档内容或文件名提取法律名称

        提取策略：
          1. 优先取文档首行（去除 Markdown 标题符号后）
          2. 首行满足"5~50字且含法律关键词"则视为法律名称
          3. 否则使用文件名（去除数字和特殊字符）
        """
        lines = content.strip().split('\n')
        if not lines:
            return stem

        # 去除 Markdown 标题符号（#、*、空格）后取首行
        first_line = re.sub(r'^[#\s*]+', '', lines[0]).strip()
        # 满足长度范围且包含法律文书关键词，则认定为法律名称
        if 5 < len(first_line) < 50 and any(k in first_line for k in ('法', '条例', '规定', '办法')):
            return first_line
        # 兜底：清理文件名中的无意义字符（数字、横线、下划线等）
        return re.sub(r'[-_\s\d]+', '', stem) or stem

    @staticmethod
    def _get_article_tag(text: str) -> str:
        """
        从分块文本中提取"第X条"标签，用于元数据标注

        Returns:
            形如"第十二条"的字符串，未找到则返回空字符串
        """
        m = re.search(r'第([零一二三四五六七八九十百千]+)条', text)
        return f"第{m.group(1)}条" if m else ""
