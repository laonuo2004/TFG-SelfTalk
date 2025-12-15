import asyncio
import json
import uuid
import traceback

import websockets
from websockets.server import WebSocketServerProtocol

import config
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from realtime_dialog_client import RealtimeDialogClient

WS_HOST = "127.0.0.1"
WS_PORT = 8765

async def ws_handler(websocket: WebSocketServerProtocol):
    session_id = uuid.uuid4().hex
    client = RealtimeDialogClient(
        config=config.ws_connect_config,
        session_id=session_id,
        output_audio_format="pcm_s16le",
        mod="audio",
        recv_timeout=30,
    )

    async def send_json(obj):
        await websocket.send(json.dumps(obj, ensure_ascii=False))

    recv_task = None
    try:
        await send_json({"type": "log", "message": f"connected, session_id={session_id}"})
        await client.connect()

        async def upstream_loop():
            while True:
                data = await client.receive_server_response()
                # 先透传，前端会显示 UPSTREAM
                await send_json({"type": "upstream", "data": data})

        recv_task = asyncio.create_task(upstream_loop())

        async for msg in websocket:
            if isinstance(msg, (bytes, bytearray)):
                await client.task_request(bytes(msg))
                continue

            payload = json.loads(msg)
            if payload.get("type") == "text":
                await client.chat_text_query(payload.get("text", ""))
            elif payload.get("type") == "start":
                await send_json({"type": "log", "message": "recording start"})
            elif payload.get("type") == "stop":
                await send_json({"type": "log", "message": "recording stop"})
            else:
                await send_json({"type": "log", "message": f"unknown: {payload}"})

    except Exception as e:
        traceback.print_exc()
        try:
            await send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            if recv_task:
                recv_task.cancel()
        except Exception:
            pass
        try:
            await client.close()
        except Exception:
            pass

async def main():
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT, max_size=10 * 1024 * 1024):
        print(f"[WS] listening on ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()

def run():
    asyncio.run(main())
