from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Any
from langchain_core.language_models import BaseChatModel
from LLM_RAG_Master.recaller.vdb.MyMilvus import MyMilvus
from LLM_RAG_Master.recaller.recaller import ReCaller, SearchType

class RagService:
    def __init__(self, search_type=SearchType.SIMILARITY.value):
        print("RagService initialised")
        self.search_type = search_type
        self.recaller = ReCaller()

    @staticmethod
    def load(file_path, using_zh_title_enhance: bool = True) -> List[Document]:
        from LLM_RAG_Master.loader.loader import DocLoader
        docs = DocLoader().get_docs_batch(file_path=file_path, using_zh_title_enhance=using_zh_title_enhance)
        return docs

    def filter(self):
        """清洗, 分割"""
        pass

    def transformer(self):
        """转化"""
        pass

    @staticmethod
    def embedding(embedding_model: HuggingFaceEmbeddings, docs: List[Document], collection_name: str, **kwargs: Any):
        """向量化，入库"""
        text_page_content = [d.page_content for d in docs]
        text_meta_data = [d.metadata for d in docs]
        MyMilvus.from_texts(
            texts=text_page_content,
            embeddings=embedding_model,
            metadatas=text_meta_data,
            collection_name=collection_name,
            **kwargs
        )

    def recall(self, query):
        """召回文档"""
        docs = self.recaller.search_docs_with_search_type(query=query, search_type=self.search_type)
        return docs

    def create_vdb_retriever(self, top_k=6, **kwargs):
        vectordb = self.recaller.get_vectordb(**kwargs)
        retriever = vectordb.as_retriever(search_type=self.search_type, search_kwargs={"k": top_k})
        return retriever

    def create_multi_query_retriever(self, llm: BaseChatModel):
        from langchain.retrievers.multi_query import MultiQueryRetriever
        retriever = self.create_vdb_retriever()
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=llm
        )
        return multi_query_retriever
