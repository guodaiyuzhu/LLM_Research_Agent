from LLM_RAG_Master.loader.embedding.embedding_model import get_m3e_embeddings
from MyMilvus import MyMilvus

embedding = get_m3e_embeddings()

vdb = MyMilvus(embedding_function=embedding, collection_name='sql_collection')

query = 'quant是什么'

docs = vdb.similarity_search(query=query, k=2)
text_page_content = [d.page_content for d in docs]
print(text_page_content)