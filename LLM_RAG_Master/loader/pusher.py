from LLM_RAG_Master.loader.loader import DocLoader

class DocPusher:
    
    @classmethod
    def insert_txt(cls, collection_name='Ollama_Embedding_Vectors', file_path = r'/home/jerryxu/Documents/knowledge_base/rag_intro.txt', **kwargs):
        # files_path = r'/home/jerryxu/Documents/knowledge_base/rag_intro.txt'
        docs = DocLoader.get_docs_batch(file_path=file_path, using_zh_title_enhance=False)
        DocLoader().insert_docs_to_vector_store_batch(docs=docs, collection_name=collection_name, **kwargs)

    @classmethod
    def insert_tables(cls, collection_name='m3e_Embedding_Index_00001', **kwargs):
        docs = DocLoader.load_tales_to_docs()
        DocLoader().insert_docs_to_vector_store_batch(docs=docs, collection_name=collection_name, **kwargs)
        print(f'Insert tables success, collection_name={collection_name}')

    @classmethod
    def insert_sql_example(cls, collection_name='m3e_Embedding_Index_SQL_01', **kwargs):
        docs = DocLoader.load_docs_from_sql_example()
        DocLoader().insert_docs_to_vector_store_batch(docs=docs, collection_name=collection_name, **kwargs)
        print(f'Insert SQL example success, collection_name={collection_name}')

if __name__ == '__main__':
    collection_name = 'rag_intro'
    DocPusher.insert_txt(collection_name=collection_name)
