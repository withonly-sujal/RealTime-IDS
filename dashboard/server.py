"""
FastAPI server — serves the dashboard UI and provides a WebSocket endpoint
for streaming real-time IDS events to the frontend.
"""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dashboard.events import EventBus
from dashboard.chat import ChatAgent


# Paths
STATIC_DIR = Path(__file__).parent / "static"


event_bus = EventBus()

chat_agent = ChatAgent(event_bus)

class ChatRequest(BaseModel):
    message: str
    session_id: str


def create_app(lifespan=None):

    app = FastAPI(title="Real-Time IDS Dashboard", lifespan=lifespan)

    @app.get("/")
    async def serve_dashboard():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.post("/api/chat")
    async def chat_endpoint(request: ChatRequest):
        reply = chat_agent.query(request.message, request.session_id)
        return {"reply": reply, "session_id": request.session_id}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """
        WebSocket endpoint — streams IDS events to the connected client.
        Each client gets its own subscriber queue from the EventBus.
        """
        await websocket.accept()
        queue = event_bus.subscribe()

        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)

        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            event_bus.unsubscribe(queue)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app
