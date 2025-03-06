from langchain_core.language_models import BaseChatModel
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import json
import os
from LLM_QueryData_Base.core.ControllerService import ControllerService

class LLMEngineWSApi:
    @staticmethod
    def start_server(controller: ControllerService, llm: BaseChatModel, host='0.0.0.0', port=17130):
        app = FastAPI()

        # 允许跨域请求
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 这里可以根据需要调整允许的来源
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

        @app.get("/")
        async def index_root():
            current_directory = os.getcwd()
            print("current_directory: " + current_directory)
            html_path = current_directory + "/LLM_QueryData_Base/src/index.html"
            with open(html_path, "r", encoding="utf-8") as file:
                html_content = file.read()
            return HTMLResponse(content=html_content)

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            client_ip = websocket.client.host

            # 设置超时时间 (单位：秒)
            timeout_seconds = 600
            while True:
                try:
                    # 接收客户端发送的消息
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=timeout_seconds)
                    
                    # 用LangChain处理问答逻辑
                    async for chunk in llm.astream(data):
                        await websocket.send_text(str(chunk.content))
                except asyncio.TimeoutError:
                    # 如果超时，关闭WebSocket连接
                    await websocket.close()
                    print(f"WebSocket连接超时，关闭连接 {client_ip}")
                    break
                except Exception as e:
                    print(f"WebSocket连接发生错误 {client_ip}: {e}")
                    break

        @app.websocket("/wsTest")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            client_ip = websocket.client.host
            # 设置超时时间 (单位：秒)
            timeout_seconds = 600
            while True:
                try:
                    # 接收客户端发送的消息
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=timeout_seconds)
                    print(f"data: {data}")
                    
                    # 进行内容识别和工具调用逻辑
                    async for chunk in controller.predict(query=data, llm=llm):
                        text = json.dumps(chunk)
                        print("text: " + text)
                        chunk_json = {"type": "websocket.send", "text": text}
                        await websocket.send(chunk_json)

                except asyncio.TimeoutError:
                    # 如果超时，关闭WebSocket连接
                    await websocket.close()
                    print(f"WebSocket连接超时，关闭连接 {client_ip}")
                    break

                except Exception as e:
                    print(f"WebSocket连接发生错误 {client_ip} : {e}")
                    break

        uvicorn.run(app, host=host, port=port)


