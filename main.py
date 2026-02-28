"""
FastAPI 主应用 — AI 法律顾问后端服务

API 路由总览：
  GET  /                      → 返回前端 index.html 页面
  GET  /api/status            → 查询系统状态（RAG 引擎、LLM、文档数等）
  POST /api/query             → 非流式问答
  POST /api/query/stream      → 流式问答（SSE）
  POST /api/upload            → 上传法律文档
  GET  /api/laws              → 获取已加载的法律列表
  DELETE /api/laws/{law_name} → 删除指定法律的所有向量片段
  DELETE /api/rebuild         → 后台重建向量库
  GET  /api/health            → 健康检查（用于容器探针等）
"""
import os
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse

from config import settings
from models import QueryRequest, QueryResponse, UploadResponse, StatusResponse
from rag_engine import rag_engine

# 配置日志格式：时间 + 级别 + 模块名 + 消息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

# 允许上传的文件格式白名单
ALLOWED_EXT = {'.pdf', '.docx', '.doc', '.pptx', '.xlsx', '.txt', '.md'}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    FastAPI 生命周期管理器（替代已废弃的 on_event）

    应用启动时自动初始化 RAG 引擎（加载嵌入模型 + 向量库 + LLM）。
    初始化失败时记录错误但不中断启动，服务以降级模式运行：
      - 查询类接口返回 503 Service Unavailable
      - 其他接口正常工作
    """
    try:
        rag_engine.initialize()
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}", exc_info=True)
        # 降级模式：is_initialized=False，后续请求会通过 503 提示用户
    yield  # 应用运行期间在此挂起，yield 后的代码在关闭时执行（此处暂无清理逻辑）


# ========== 应用实例 ==========
app = FastAPI(title="AI 法律顾问", version="1.0.0", lifespan=lifespan)

# 跨域中间件：允许前端（如 React 开发服务器）跨域访问后端 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # 生产环境建议改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 将当前目录挂载为静态文件服务，用于提供前端资源（JS/CSS 等）
frontend_path = Path(__file__).parent
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# ========== 路由定义 ==========

@app.get("/", include_in_schema=False)
async def root():
    """根路径：返回前端 SPA 页面（index.html），不存在则返回 JSON 状态"""
    f = frontend_path / "index.html"
    return FileResponse(str(f)) if f.exists() else JSONResponse({"status": "running"})


@app.get("/api/status", response_model=StatusResponse)
async def status():
    """系统状态接口：返回 RAG 引擎配置、已加载法律列表、LLM 信息等"""
    return rag_engine.get_status()


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    非流式问答接口：一次性返回完整答案

    适用场景：对延迟不敏感、需要结构化响应解析的客户端
    """
    if not rag_engine.is_initialized:
        raise HTTPException(503, "引擎未就绪")
    try:
        return QueryResponse(**(await rag_engine.query(req.question)))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """
    流式问答接口：通过 SSE（Server-Sent Events）逐 Token 推送答案

    前端监听格式：
      data: <文本片段>\\n\\n
      data: [DONE]\\n\\n  ← 结束标志

    适用场景：需要"打字机"效果的 Web 聊天界面
    """
    if not rag_engine.is_initialized:
        raise HTTPException(503, "引擎未就绪")

    async def gen():
        # 逐 Token 包装成 SSE 格式推送
        async for chunk in rag_engine.query_stream(req.question):
            yield f"data: {chunk}\n\n"
        # 发送结束标志，前端据此关闭 EventSource 连接
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",          # 禁止代理缓存 SSE 响应
            "X-Accel-Buffering": "no",               # 禁止 Nginx 缓冲（确保实时推送）
        },
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    """
    文档上传接口：解析文件并将向量写入 ChromaDB

    处理流程：
      1. 校验文件格式是否在白名单内
      2. 将文件写入临时目录（避免污染工作目录）
      3. 调用 RAG 引擎解析并入库
      4. finally 块确保临时文件始终被清理
    """
    if not rag_engine.is_initialized:
        raise HTTPException(503, "系统未就绪")

    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    # 格式校验：拒绝不支持的文件类型
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"不支持的格式: {ext}")

    # 写入临时文件（mkdtemp 保证目录唯一，防止文件名冲突）
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)
    try:
        with open(tmp_path, 'wb') as f:
            f.write(await file.read())   # 异步读取上传内容写入磁盘
        result = await rag_engine.add_document(tmp_path)
        return UploadResponse(success=True, message=f"'{filename}' 加载成功", **result)
    except Exception as e:
        raise HTTPException(500, f"文档处理失败: {e}")
    finally:
        # 无论成功/失败，都清理临时文件和目录
        if os.path.exists(tmp_path): os.unlink(tmp_path)
        if os.path.exists(tmp_dir):  os.rmdir(tmp_dir)


@app.get("/api/laws")
async def laws():
    """获取向量库中所有已加载的法律名称及文档总数"""
    return {
        "laws":      rag_engine.law_names,
        "total":     len(rag_engine.law_names),
        "doc_count": rag_engine.doc_count,
    }


@app.delete("/api/laws/{law_name}")
async def delete_law(law_name: str):
    """
    按法律名称删除向量库中对应的所有片段

    Args:
        law_name: URL 路径参数，需与向量库元数据中的 law_name 完全匹配
    """
    if not rag_engine.is_initialized:
        raise HTTPException(503, "系统未就绪")
    try:
        result = await rag_engine.delete_law(law_name)
        return result
    except ValueError as e:
        # 业务逻辑错误（未找到法律、名称为空等）→ 400 Bad Request
        raise HTTPException(400, str(e))
    except Exception as e:
        # 其他未预期错误 → 500 Internal Server Error
        raise HTTPException(500, f"删除失败: {str(e)}")


@app.delete("/api/rebuild")
async def rebuild(bg: BackgroundTasks):
    """
    重建向量库（后台异步执行，避免阻塞当前请求）

    执行步骤：
      1. 删除现有 ChromaDB 目录
      2. 重置引擎初始化状态
      3. 重新扫描法律文档目录并建库

    注意：重建期间查询接口将返回 503，建议在低峰期操作
    """
    def _do():
        import shutil
        # 删除持久化的向量数据库目录
        if os.path.exists(settings.CHROMA_DB_PATH):
            shutil.rmtree(settings.CHROMA_DB_PATH)
        rag_engine.is_initialized = False  # 标记为未初始化，触发服务降级
        rag_engine.initialize()            # 重新初始化（重新扫描法律文档）

    # 将重建任务加入后台队列，当前请求立即返回
    bg.add_task(_do)
    return {"message": "后台重建中"}


@app.get("/api/health")
async def health():
    """
    轻量健康检查接口

    适用场景：Docker 健康探针、负载均衡器心跳检测
    仅检查应用进程是否存活及 RAG 引擎初始化状态，不做深度检查
    """
    return {"status": "ok", "initialized": rag_engine.is_initialized}


# ========== 开发模式入口 ==========
if __name__ == "__main__":
    import uvicorn
    # reload=True 开启热重载，代码修改后自动重启（仅用于开发，生产环境请去掉）
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
