#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小可用的 WebSocket 中继服务：
- 浏览器发来的二进制 PCM (S16LE/16k/mono) → 转发到火山 RealtimeDialog
- 火山返回的 TTS PCM → 原样二进制下发给浏览器播放
- 文本/控制用 JSON 文本通道：{"type":"control","op":"start|stop|text","text": "..."}
依赖：
  pip install websockets
要求同目录存在：
  - config.py
  - realtime_dialog_client.py（内部依赖 protocol.py）
"""
import asyncio
import json
import uuid
from typing import Dict, Any

import websockets
from websockets.server import WebSocketServerProtocol

import config
from realtime_dialog_client import RealtimeDialogClient


class DialogSession:
    """
    一条浏览器 WebSocket 连接对应一个火山对话会话。
    """
    def __init__(
        self,
        ws_config: Dict[str, Any],
        output_audio_format: str = "pcm_s16le",
        mod: str = "audio",
        recv_timeout: int = 10,
        expect_sample_rate: int = 16000,
    ):
        self.session_id = str(uuid.uuid4())
        self.output_audio_format = output_audio_format
        self.mod = mod
        self.recv_timeout = recv_timeout
        self.expect_sample_rate = expect_sample_rate

        self.recording_enabled = False
        # 20ms 一帧：16k * 0.02 * 2 bytes = 640 bytes（小于此丢掉，防止碎片太小）
        self.min_chunk_bytes = 640

        # 火山实时对话客户端（沿用你的现有实现）
        self.client = RealtimeDialogClient(
            config=ws_config,
            session_id=self.session_id,
            output_audio_format=output_audio_format,
            mod=mod,
            recv_timeout=recv_timeout,
        )

    async def _send_json(self, websocket: WebSocketServerProtocol, obj: Dict[str, Any]):
        try:
            await websocket.send(json.dumps(obj, ensure_ascii=False))
        except Exception as e:
            print(f"[DialogSession] send_json error: {e}")

    async def handle_server_response(self, response: Dict[str, Any], websocket: WebSocketServerProtocol):
        """
        处理火山侧返回：
        - SERVER_ACK：payload_msg 为 PCM 字节，直接二进制发给前端
        - SERVER_FULL_RESPONSE：事件/文本，转成 JSON 文本下发
        - SERVER_ERROR：错误转成 JSON
        """
        if not response:
            return

        msg_type = response.get("message_type")

        if msg_type == "SERVER_ACK" and isinstance(response.get("payload_msg"), (bytes, bytearray)):
            await websocket.send(response["payload_msg"])

        elif msg_type == "SERVER_FULL_RESPONSE":
            await self._send_json(websocket, {
                "type": "event",
                "event": response.get("event"),
                "payload": response.get("payload_msg"),
            })

        elif msg_type == "SERVER_ERROR":
            await self._send_json(websocket, {"type": "error", "message": str(response.get("payload_msg"))})
            raise Exception(f"服务器错误：{response.get('payload_msg')}")

    async def run(self, websocket: WebSocketServerProtocol):
        print(f"[DialogSession] start session_id={self.session_id}")
        try:
            # 1) 建连火山
            await self.client.connect()

            # 2) 可选：打个招呼（你现有实现里若无该方法，会被 try 捕获忽略）
            try:
                if hasattr(self.client, "say_hello"):
                    await self.client.say_hello()
            except Exception as e:
                print(f"[DialogSession] say_hello failed: {e}")

            # 前端→模型
            async def frontend_to_model():
                try:
                    async for message in websocket:
                        # 二进制：音频帧
                        if isinstance(message, (bytes, bytearray)):
                            if self.recording_enabled and message and len(message) >= self.min_chunk_bytes:
                                await self.client.task_request(bytes(message))
                            continue

                        # 字符串：控制/文本
                        text = (message or "").strip()
                        obj = None
                        try:
                            obj = json.loads(text)
                        except Exception:
                            pass

                        if isinstance(obj, dict) and obj.get("type") == "control":
                            op = obj.get("op")
                            if op == "start":
                                self.recording_enabled = True
                                await self._send_json(websocket, {"type": "control_ack", "op": "start"})
                            elif op == "stop":
                                self.recording_enabled = False
                                await self._send_json(websocket, {"type": "control_ack", "op": "stop"})
                            elif op == "text":
                                q = (obj.get("text") or "").strip()
                                if q:
                                    await self.client.chat_text_query(q)
                            else:
                                await self._send_json(websocket, {"type": "warn", "message": f"未知操作: {op}"})
                            continue

                        # 兜底：当作纯文本 query
                        if text:
                            await self.client.chat_text_query(text)

                except websockets.exceptions.ConnectionClosed:
                    print(f"[DialogSession] websocket closed by frontend (send), session={self.session_id}")
                except Exception as e:
                    print(f"[DialogSession] frontend_to_model error: {e}")

            # 模型→前端
            async def model_to_frontend():
                try:
                    while True:
                        response = await self.client.receive_server_response()
                        await self.handle_server_response(response, websocket)
                except websockets.exceptions.ConnectionClosed:
                    print(f"[DialogSession] websocket closed by frontend (recv), session={self.session_id}")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"[DialogSession] model_to_frontend error: {e}")

            # 并发跑
            task_send = asyncio.create_task(frontend_to_model())
            task_recv = asyncio.create_task(model_to_frontend())
            done, pending = await asyncio.wait([task_send, task_recv], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()

        finally:
            print(f"[DialogSession] closing RealtimeDialog session {self.session_id}")
            # 顺序关闭
            for fn in (self.client.finish_session, self.client.finish_connection, self.client.close):
                try:
                    await fn()
                except Exception as e:
                    print(f"[DialogSession] close error: {e}")


# ========== WebSocket 服务端入口 ==========
async def ws_handler(websocket: WebSocketServerProtocol, path: str):
    print(f"[WS] new frontend connection, path={path}")
    session = DialogSession(
        ws_config=config.ws_connect_config,
        output_audio_format="pcm_s16le",
        mod="audio",
        recv_timeout=10,
        expect_sample_rate=16000,
    )
    await session.run(websocket)
    print(f"[WS] frontend connection closed, path={path}")


async def main():
    # 改成你机器的实际侦听地址/端口；线上建议走 wss 并配置证书反向代理
    host, port = "0.0.0.0", 6006
    server = await websockets.serve(
        ws_handler,
        host,
        port,
        max_size=4 * 1024 * 1024,  # 允许足够的帧
        ping_interval=20,
        ping_timeout=20,
    )
    print(f"[WS] relay server listening on {host}:{port}")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
