from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import asyncio
from LLM_ChatIntent_Pro.ChatIntentService import ChatIntentService

class IntentWsApi:
    @staticmethod
    def start_server(host='0.0.0.0', port=17180):
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/")
    async def index_root():
        return "hello hello"

    @app.get("/chatIntent/query={text}")
    def intent_detect(text: str):
        current_directory = os.getcwd()
        intent_model_path = f"{current_directory}/"
        chat_intent = ChatIntentService(model_path=intent_model_path)
        res = asyncio.run(chat_intent.detect(text))['intent']
        print(text)
        print(res)
        return res
    uvicorn.run(app, host=host, port=port)

IntentWsApi.start_server()

