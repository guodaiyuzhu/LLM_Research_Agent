from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from LLM_RAG_Master.config.config import *

def get_ollama_embeddings(model=OLLAMA_DEFAULT_MODEL, base_url=OLLAMA_BASE_URL, **kwargs):
    embedding = OllamaEmbeddings(model=model, temperature=0, base_url=base_url, **kwargs)
    return embedding

def get_m3e_embeddings(model=EMBEDDING_MODEL, model_kwargs=None, **kwargs):
    if model_kwargs is None:
        model_kwargs = {'device': EMBEDDING_DEVICE}
    embedding = HuggingFaceEmbeddings(model_name=model, model_kwargs=model_kwargs, **kwargs)
    return embedding
