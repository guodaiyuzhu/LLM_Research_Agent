from LLM_RAG_Master.recaller.vdb.MyMilvus import MyMilvus
from LLM_RAG_Master.loader.embedding.embedding_model import get_m3e_embeddings
from LLM_RAG_Master.recaller.baseSearch import BaseSearch
from enum import Enum

class SearchType(Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"

class ReCaller(BaseSearch):
    """召回类"""

    def __init__(self):
        self.embedding_model = get_m3e_embeddings()
        print("ReCaller initialized")

    def get_vectordb(self, **kwargs):
        embedding = self.embedding_model
        vdb = MyMilvus(embedding_function=embedding, **kwargs)
        return vdb

    def base_search(self, query: str, **kwargs):
        vdb = self.get_vectordb(**kwargs)
        docs = vdb.similarity_search(query)
        return docs

    def search_docs_with_search_type(self, query: str, search_type: str, **kwargs):
        if search_type == SearchType.SIMILARITY.value:
            docs = self.base_search(query, **kwargs)
        else:
            docs = []
        return docs

    def search_from_es(self):
        """ES 召回项"""
        pass

    def retriever_from_prompt(self):
        pass
