"""FastAPI 后端：提供 Web 可视化页面 + SSE 流式聊天接口。"""

import json
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.agent.research_agent import ResearchAgent


app = FastAPI(title="智能研究助手")

# 服务端全局 Agent 单例，保留多轮记忆
_agent = ResearchAgent()

# 静态资源
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    """返回聊天页面。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/chat")
async def chat(q: str):
    """SSE 流式聊天接口。前端用 EventSource 监听。

    事件流格式：
        data: {"chunk": "你"}     ← 文本片段
        data: {"chunk": "好"}
        data: {"done": true}     ← 结束标志
    """
    if not q.strip():
        return JSONResponse({"error": "问题不能为空"}, status_code=400)

    def event_stream() -> Iterator[str]:
        try:
            for chunk in _agent.chat_stream(q):
                payload = json.dumps({"chunk": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except Exception as e:
            err = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"
        finally:
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 nginx 缓冲，确保流式
        },
    )


@app.post("/reset")
async def reset() -> JSONResponse:
    """清空当前会话记忆。"""
    _agent.reset()
    return JSONResponse({"ok": True})
