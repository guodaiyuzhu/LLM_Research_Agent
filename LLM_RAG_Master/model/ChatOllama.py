# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
class ChatOllamaModel:

    @classmethod
    def chatOllama(cls, model: str='qwen2:7b',base_url="http://localhost:11434",
                   temperature=0,
                   top_p=0.75,
                   **kwargs):
        llm = ChatOllama(model=model,
                         base_url=base_url,
                         temperature=temperature,
                         top_p=top_p,
                         **kwargs
                         )
        return llm