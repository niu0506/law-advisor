"""
FastAPI 主应用 — AI 法律顾问
"""
import os, logging, tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse

from config import settings
from models import QueryRequest, QueryResponse, UploadResponse, StatusResponse
from rag_engine import rag_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_EXT = {'.pdf', '.docx', '.doc', '.pptx', '.xlsx', '.txt', '.md'}


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        rag_engine.initialize()
    except Exception as e:
        logger.error(f"❌ 启动失败: {e}", exc_info=True)
        # 注意: 引擎未初始化，服务将以降级模式运行，查询接口将返回503
    yield


app = FastAPI(title="AI 法律顾问", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

frontend_path = Path(__file__).parent
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    f = frontend_path / "index.html"
    return FileResponse(str(f)) if f.exists() else JSONResponse({"status": "running"})


@app.get("/api/status", response_model=StatusResponse)
async def status():
    return rag_engine.get_status()


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not rag_engine.is_initialized:
        raise HTTPException(503, "引擎未就绪")
    try:
        return QueryResponse(**(await rag_engine.query(req.question)))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    if not rag_engine.is_initialized:
        raise HTTPException(503, "引擎未就绪")

    async def gen():
        async for chunk in rag_engine.query_stream(req.question):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if not rag_engine.is_initialized:
        raise HTTPException(503, "系统未就绪")
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"不支持的格式: {ext}")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)
    try:
        with open(tmp_path, 'wb') as f:
            f.write(await file.read())
        result = await rag_engine.add_document(tmp_path)
        return UploadResponse(success=True, message=f"'{filename}' 加载成功", **result)
    except Exception as e:
        raise HTTPException(500, f"文档处理失败: {e}")
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)
        if os.path.exists(tmp_dir): os.rmdir(tmp_dir)


@app.get("/api/laws")
async def laws():
    return {"laws": rag_engine.law_names, "total": len(rag_engine.law_names), "doc_count": rag_engine.doc_count}


@app.delete("/api/laws/{law_name}")
async def delete_law(law_name: str):
    if not rag_engine.is_initialized:
        raise HTTPException(503, "系统未就绪")
    try:
        result = await rag_engine.delete_law(law_name)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"删除失败: {str(e)}")


@app.delete("/api/rebuild")
async def rebuild(bg: BackgroundTasks):
    def _do():
        import shutil
        if os.path.exists(settings.CHROMA_DB_PATH): shutil.rmtree(settings.CHROMA_DB_PATH)
        rag_engine.is_initialized = False
        rag_engine.initialize()
    bg.add_task(_do)
    return {"message": "后台重建中"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "initialized": rag_engine.is_initialized}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
