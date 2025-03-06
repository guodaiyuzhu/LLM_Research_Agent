from typing import Optional
from langchain_community.document_loaders.text import TextLoader

from LLM_RAG_Master.RAGService import RagService
from LLM_RAG_Master.config.config import (CHUNK_SIZE, CHUNK_OVERLAP, ZH_TITLE_ENHANCE, DEFAULT_MILVUS_CONNECTION)
from LLM_RAG_Master.Utils.my_text_splitter import MyTextSplitter
from LLM_RAG_Master.Utils.utils import tree
from LLM_RAG_Master.Utils.ConvertWORDtoTXT import ConvertWordToTxt
from langchain.schema import Document
from LLM_RAG_Master.recaller.vdb.MyMilvus import MyMilvus
from LLM_RAG_Master.loader.embedding.embedding_model import get_m3e_embeddings

class DocLoader:
    embedding_model = get_m3e_embeddings()

    @staticmethod
    def zh_title_enhance(docs):
        """Enhance chunk titles"""
        if len(docs) > 0:
            title = docs[0].page_content
            for i in range(1, len(docs)):
                docs[i].page_content = f"标题: {title} 内容: {docs[i].page_content}"
            return docs
        else:
            print("文件不存在")

    @staticmethod
    def get_docs_from_file(file_absolute_path):
        """Load documents from file using LangChain document loaders"""
        docs = []
        if file_absolute_path.lower().endswith(".txt"):
            loader = TextLoader(file_absolute_path, autodetect_encoding=True)
            documents = loader.load()
            texts = documents[0].page_content
            output_texts = MyTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_text(texts)
            for output_text in output_texts:
                docs.append(Document(page_content=output_text, metadata={"source": {"filepath":f"{file_absolute_path}", "text": f"{output_text}"}}))
        elif file_absolute_path.lower().endswith(".docx"):
            text_splitter = MyTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            output_texts = ConvertWordToTxt(file_absolute_path, text_splitter)
            for output_text in output_texts:
                docs.append(Document(page_content=output_text[1], metadata={"source": f"{file_absolute_path}", "title": output_text[0]}))
        return docs

    @classmethod
    def get_docs_from_file_with_extract_words(cls, file_absolute_path, wiki):
        """Load documents and extract words"""
        docs = []
        # if file_absolute_path.lower().endswith(".txt"):
        #     loader = TextLoader(file_absolute_path, autodetect_encoding=True)
        #     documents = loader.load()
        #     texts = [doc.page_content for doc in documents]
        #     output_texts = MyTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_text(texts)
        #     for output_text in output_texts:
        #         keywords = extract_words(output_text)
        #         docs.append(Document(page_content=keywords, metadata={"source": {"filepath":f"{file_absolute_path}", "text": f"{output_text}"}}))
        # elif file_absolute_path.lower().endswith(".docx"):
        #     text_splitter = MyTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        #     output_texts = ConvertWordToTxt(file_absolute_path, text_splitter)
        #     for output_text in output_texts:
        #         keywords = extract_words(output_text)
        #         if keywords=='':
        #             keywords=output_text
        #         docs.append(Document(page_content=keywords, metadata={"source": {"filepath":f"{file_absolute_path}", "text": f"{output_text}"}}))
        return docs

    @classmethod
    def get_docs_batch(cls, file_path, using_zh_title_enhance=ZH_TITLE_ENHANCE):
        file_absolute_path_list = tree(file_path)
        loaded_files = []
        failed_files = []
        docs=[]
        for i in range(0, len(file_absolute_path_list), 1000):
            file_absolute_path_list_batch = file_absolute_path_list[i:i + 1000]
            for file_absolute_path in file_absolute_path_list_batch:
                try:
                    doc = cls.get_docs_from_file(file_absolute_path)
                    if using_zh_title_enhance:
                        doc = cls.zh_title_enhance(doc)
                    docs+=doc
                    print(f"已成功加载：{file_absolute_path}")
                    loaded_files.append(file_absolute_path)
                except Exception as e:
                    print(f"未能成功加载：{file_absolute_path}, {e}")
                    failed_files.append(file_absolute_path)
        return docs

    @staticmethod
    def insert_docs_to_vector_store_batch(collection_name, embedding=embedding_model, connection_args=DEFAULT_MILVUS_CONNECTION, docs:Optional[list[Document]]=None, **kwargs):
        """Insert docs into a vector store batch"""
        for i in range(0, len(docs), 1000):
            docs_batch = docs[i:i + 1000]
            my_milvus = MyMilvus(embedding_function=embedding, collection_name=collection_name, connection_args=connection_args, **kwargs)
            text_page_content = [d.page_content for d in docs_batch]
            text_meta_data = [d.metadata for d in docs_batch]
            my_milvus.insert_to_vdb(texts=text_page_content, embeddings=embedding, metadatas=text_meta_data)


    @staticmethod
    def load_tales_to_docs():
        # Oracle database connection for loading tables into docs
        # import cx_Oracle
        #
        # conn = cx_Oracle.connect('ficc_dwods/LHAK_lqpg_326@168.63.18.146:1521/heads')
        # cursor = conn.cursor()
        # sql = "SELECT TABLE_NAME, COMMENTS FROM USER_TAB_COMMENTS WHERE COMMENTS IS NOT NULL AND TABLE_TYPE = 'TABLE'"
        # cursor.execute(sql)
        # tables=cursor.fetchall()
        tables_json = {
            "FICC_TRADE": "FICC 交易类文档数据",
            "FICC_ORDER": "FICC 订单类文档数据",
            "FICC_POSITION": "FICC 持仓类文档数据",
            "FICC_RDS_BOND": "FICC RDS 债券数据",
            "FICC_RDS_BOND_VALUATION":"RDS 债券基础信息",
            "FICC_PORT_PNL": "FICC组合损益计量",
            "FICC_PORT_RISK": "FICC组合风险计量"
            # Add other table descriptions as required
        }

        tables = list(tables_json.keys())
        tables_info_list = []

        for table in tables:
            table_info = (f"Table = FICC_DWODS.{table}; description = {tables_json[table]}; columns=[")
            sql_query = f"SELECT * FROM ods_col_test t1 WHERE t1.table_name = '{table}'"
            cursor.execute(sql_query)
            sql_query_results = cursor.fetchall()
            columns = ''

            for r in sql_query_results:
                columns += f"{table}.{r[1]}: {r[2] if r[2] else 'None'},"
            table_info += columns+']'
            table_info_list = [tables_json[table], table_info]
            tables_info_list.append(table_info_list)

        docs = []
        for comments, doc in tables_info_list:
            docs.append(Document(page_content=comments, metadata={"source": doc}))
        return docs

    # Loading docs from SQL example
    @classmethod
    def load_docs_from_sql_example(cls):
        docs = []
        sql_examples = [
            {
                "Human": "查询ZSJY08_01账号近3天的持仓总金额",
                "AI": {
                    "SQL_Scripts": "SELECT SUM(t.current_holding_cost) FROM ficc_dwods.ficc_position t WHERE t.book_name = 'ZSJY08_01' AND t.trade_date >= TO_CHAR(SYSDATE - 3, 'YYYYMMDD') AND 1=1"
                }
            },
            {
                "Human": "查询BOOKTRADER03账号再20240601到20240604的持仓总金额",
                "AI": {
                    "SQL_Scripts": "SELECT SUM(t.current_holding_cost) FROM ficc_dwods.ficc_position t WHERE t.book_name = 'BOOKTRADER03' AND t.trade_date >='20240601' "
                    "AND t.trade_date >='20240604' AND 1=1"
                }
            }
        ]
        for i in sql_examples:
            docs.append(Document(page_content=i["Human"], metadata={"source": i}))
        return docs

# Load and insert data into vector DB
def load_poc_sql_to_docs():
    # import cx_Oracle
    # conn = cx_Oracle.connect('ficc_dwods/LHAK_lqpg_326@168.63.18.146:1521/heads')
    # cursor = conn.cursor()
    # sql = "SELECT TABLE_NAME, COMMENTS FROM USER_TAB_COMMENTS WHERE COMMENTS IS NOT NULL AND TABLE_TYPE = 'TABLE'"
    # cursor.execute(sql)
    # tables = cursor.fetchall()

    tables_json = {
        "quant": "SELECT * FROM ficc_quant LIMIT 10",
        "position": "SELECT * FROM ficc_position LIMIT 10",
        "risk": "SELECT * FROM ficc_risk LIMIT 10",
    }

    tables = list(tables_json)
    tables_info_list = []

    for table in tables:
        table_info = f"帮我查询 {table} 相关信息"
        table_info_list = [tables_json[table], table_info]
        tables_info_list.append(table_info_list)

    docs = []
    for comments, doc in tables_info_list:
        docs.append(Document(page_content=comments, metadata={"source": doc}))
    return docs

if __name__=='__main__':
    collection_name = 'sql_collection'
    # insert data to vector db
    # docs = load_poc_sql_to_docs()
    # try:
    #     DocLoader().insert_docs_to_vector_store_batch(docs=docs, collection_name=collection_name)
    # except Exception as e:
    #     print(e)
    # Query data from vector db
    query = "帮我看看quant表的相关信息"
    rag_service = RagService()
    retriever = rag_service.create_vdb_retriever(collection_name=collection_name, top_k=2)
    example_retriever_response = retriever.invoke(query)
    print(example_retriever_response)

        #
        # from pymilvus import connections, db
    # conn = connections.connect(host="192.168.2.105", port=19530)
    # database = db.create_database("my_database")