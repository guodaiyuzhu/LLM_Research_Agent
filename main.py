import os  
from LLM_QueryData_Base.api.LLM_Engine_Ws_Api import LLMEngineWSApi
from LLM_QueryData_Base.core.ControllerService import ControllerService  
from LLM_RAG_Master.model.ChatOllama import ChatOllamaModel

current_directory = os.getcwd()
chat_intent_model_path = f"{current_directory}/LLM_ChatIntent_Pro/"
controller = ControllerService(chat_intent_model_path)
llm = ChatOllamaModel.chatOllama()
LLMEngineWSApi.start_server(controller, llm, port=17130)
