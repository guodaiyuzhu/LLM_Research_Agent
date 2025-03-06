from typing import List, Optional, Any
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from LLM_RAG_Master.config.config import *
from LLM_RAG_Master.loader.embedding.embedding_model import get_m3e_embeddings
from pymilvus import connections, Collection, RRFRanker, WeightedRanker


class MyMilvus(Milvus):
    
    def __init__(self, embedding_function: Embeddings, 
                 collection_name: Optional[str] = DEFAULT_COLLECTION_NAME,
                 connection_args: Optional[dict[str, Any]] = DEFAULT_MILVUS_CONNECTION,
                 search_params: Optional[dict[str, Any]] = DEFAULT_SEARCH_PARAMS,
                 index_params: Optional[dict[str, Any]] = DEFAULT_INDEX_PARAMS,
                 **kwargs):
        super().__init__(
            embedding_function=embedding_function,
            collection_name=collection_name,
            connection_args=connection_args,
            search_params=search_params,
            index_params=index_params,
            **kwargs
        )
        self.collection_name = collection_name
        self.connection_args = connection_args
        self.search_params = search_params
        self.index_params = index_params
        self.kwargs = kwargs
    
    def similarity_search(self, query: str, k: int = 2, param=None, expr=None, **kwargs: Any) -> (List)[Document]:
        return super().similarity_search(query, k, param=self.search_params, expr=expr, **kwargs)
    
    def insert_to_vdb(self, 
                      texts: List[str], 
                      embeddings: Embeddings, 
                      metadatas: Optional[List[dict]] = None, 
                      **kwargs: Any):
        super().from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            connection_args=self.connection_args,
            collection_name=self.collection_name,
            index_params=self.index_params,
            **kwargs
        )

    def __call__(self, *args, **kwargs):
        return MyMilvus

def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
    return super().as_retriever(**kwargs)

def mul_vector_search(data):
    lst = data.split()
    query_list = [[lst[0], 'class'], [lst[1], 'subclass'], [lst[2], 'text']]
    
    connections.connect(
        host="127.0.0.1",  # IP address of Milvus server
        port="19530"
    )
    
    collection = Collection(name="Ollama_Embedding_Vectors_Test04")
    collection.load()
    
    reqs = []
    for i in query_list:
        query_vector = get_m3e_embeddings().embed_query(i[0])
        search_param = {
            "data": [query_vector],  # Query vector
            "anns_field": i[1],  # Field to search
            "param": {
                "metric_type": "L2",  # Distance metric
                "params": {"nprobe": 10}
            },
            "limit": 2  # Limit the number of results
        }
        request = AnnSearchRequest(**search_param)
        reqs.append(request)
    
    rerank = RRFRanker()
    
    # You can also use a weighted ranker if needed:
    # rerank = WeightedRanker(0.6, 0.6, 0.9)
    
    res = collection.hybrid_search(
        reqs,  # List of AnnSearchRequest objects
        rerank,  # Rerank strategy
        # output_fields = ['source'],
        limit=2  # Limit the number of results
    )
    
    res_list = []
    for i in res[0]:
        search_param = {
            "output_fields": ["content"],
            "expr":"id==i.id",
        }
        res_list.append(collection.query('id=='+str(i.id),["content"])[0]['content'])
    return res_list

