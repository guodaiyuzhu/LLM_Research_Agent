from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from FICC_RAG_Master.RagService import RagService
from FICC_RAG_Master.config.config import DEFAULT_COLLECTION_NAME
from LLM_QueryData_Base.log.logutil import my_logger

knowledge_template = """
You are an assistant for question-answering tasks.
You can reference or not the following pieces of retrieved context and Chat History to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Chat History: {history}
Question: {question}
Context: {context}
Answer:
"""


class KnowledgeService:
    def __init__(self):
        self.rag_service = RagService()

    def search(self, query, search_flag, **kwargs):
        docs = self.rag_service.retrieve(query=query, search_flag=search_flag, **kwargs)
        doc_contents = [i.page_content for i in docs]
        return doc_contents

    async def predict(self, query, search_flag, llm: BaseChatModel, memory, collection_name=DEFAULT_COLLECTION_NAME, **kwargs):
        doc_contents = self.search(query,search_flag,collection_name=collection_name,**kwargs)
        prompt = PromptTemplate.from_template(
            template=knowledge_template,
            partial_variable={"context":str(doc_contents), "history": str(memory.chat_memory)}
        )
        chain = prompt | llm
        async for res in chain.astream({"question": query}):
            yield {"llm": res.content}
        yield {"knowledge": str(doc_contents)}
