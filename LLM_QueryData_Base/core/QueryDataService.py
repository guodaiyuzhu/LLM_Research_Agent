import json
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
from LLM_RAG_Master.Utils.tools import get_data_from_data_api
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from LLM_RAG_Master.RAGService import RagService

system_template = """
You are an assistant that has access to the following set of tools.
Given the user input, return the name and input of the tool to use.
Return your response as a JSON blob with 'name' and 'arguments' keys.

Here are the names and descriptions for each tool:
{tools}

User: {input}
"""

data_api_template = """
You are an AI assistant, and you have some background knowledge here.
knowledge:
{knowledge}
You need to turn the questions raised by Human into SQL scripts,
and you can refer to some examples to answer them.
examples:
{example}
Human: {input}
AI:
"""

tools_call_template = """
你是一个 AI 查询工具调用助手，目的是将用户问题转化成查询要素并且调用接口查询结果并返回给用户。
下面是工具的 INPUT 和 OUTPUT：
工具查询要素，INPUT：{sql_args}
接口查询后返回的的内容，OUTPUT：{function_call_response}
=============================================================================================================================================
接口查询返回的内容主要是获取里面的数据，返回给用户，但是也会有查询异常，因此必须严格按如下处理逻辑处理后输出给用户
<1>OUTPUT 正常返回做如下处理：
当接口返回的内容类似如下 JSON 格式：
[{{"TRADE_DATE": "20240920", "BOOK_NAME": "LHJY01_1", "INITIAL_HOLDING_QTY": 1, "ROW_ID": 1}},{{"TRADE_DATE": "20240920", "BOOK_NAME": "LHJY01_1", "INITIAL_HOLDING_QTY": 100, "ROW_ID": 2}}, {{"TRADE_DATE": "20240920", "BOOK_NAME": "LHJY01_1", "INITIAL_HOLDING_QTY": 1291, "ROW_ID": 3}}]
解析成 Markdown 格式，并在开始加前缀语：" 您好，您查询的数据如下："
|TRADE_DATE	|BOOK_NAME	|INITIAL_HOLDING_QTY	|ROW_ID  |
|20240920	|LHJY01_1	|1	                    |1       |
|20240920	|LHJY01_1	|100	                |2       |
|20240920	|LHJY01_1	|1291	                |3       |
在 Markdown 数据后再稍加一点数据指标的分析和总结描述，例如对数据的趋势，最大最小值，均值等分析。不超过 100 字。
<2>OUTPUT 返回做如下处理：
2.1 当接口返回的 OUTPUT 包含 "book name error" 或者相关内容，最后只按如下内容输出："抱歉，您给我的用户名不对，请麻烦输入大象系统录入的有效用户名，谢谢！";
2.2 当接口返回的 OUTPUT 包含 "ORA-XXX" 问题，将 ORA-XXX 后面的内容作为 ERROR_OUTPUT，最后只按如下内容输出："抱歉，查询出现下面问题：$ERROR_OUTPUT，可能是我们没沟通清楚或者查询条件，谢谢！";
2.3 当接口返回的 OUTPUT 是 "[]"，最后只按如下内容输出："抱歉，结果为空，您查询的请求未匹配到数据或者数据未录入！";
2.4 其他接口返回异常，最后只按如下内容输出："抱歉，未查询到您的问题，请换个说法或者查询条件，谢谢！";
一定注意以下要求：
1、必须严格按照上述要求输出，不得随意发挥和自由输出；
2、必须严格上述 <1> 和 < 2 > 声明的要求输出内容，除此之外，不得输出其他任何内容。
"""


class QueryDataService:
    def __init__(self, tools=None):
        if tools is None:
            tools = [get_data_from_data_api]
        self.knowledge_retriever_collection_name = 'FICC_TABLE_INFO_01'
        self.knowledge_retriever_top_k = 2
        self.example_retriever_collection_name = 'sql_collection'
        self.example_retriever_top_k = 2
        self.tools = tools
        self.render_tools = render_text_description(self.tools)
        self.rag_service = RagService()

    def generate_sql(self, query, llm: BaseChatModel):
        knowledge_retriever = self.rag_service.create_vdb_retriever(
            collection_name=self.knowledge_retriever_collection_name,
            top_k=self.knowledge_retriever_top_k
        )
        example_retriever = self.rag_service.create_vdb_retriever(
            collection_name=self.example_retriever_collection_name,
            top_k=self.example_retriever_top_k
        )
        knowledge_retriever_response = knowledge_retriever.invoke(query)
        example_retriever_response = example_retriever.invoke(query)
        knowledge_contents = ''
        example_contents = ''
        for i in range(1, len(knowledge_retriever_response) + 1, 1):
            knowledge_contents += f'{i}. {knowledge_retriever_response[i - 1].metadata["source"]}\n'
        for i in example_retriever_response:
            example_contents += f'Human: {i.content}\nAI: {i.metadata["source"]}\n\n'
        prompt = PromptTemplate(
            template=data_api_template,
            partial_variables={
                "knowledge": knowledge_contents,
                "example": example_contents,
                "input": query
            }
        )
        chain = prompt | llm
        response = chain.invoke({"input": query})
        print(f'generate_sql response: {response.content}')
        return response.content

    def tool_chain(self, model_output):
        tool_map = {tool.name: tool for tool in self.tools}
        chosen_tool = tool_map[model_output["name"]]
        return chosen_tool

    def query_data_from_sql_by_tools(self, query, model: BaseChatModel):
        sql = self.generate_sql(query, model)
        sql = "select * from ficc_position"
        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template=system_template,
            input_variables=["input"],
            partial_variables={"tools": self.render_tools}
        )
        chain = prompt | model | parser
        response = chain.invoke({"input": sql})
        print(response)
        return response


if __name__ == "__main__":
    from LLM_RAG_Master.model.ChatOllama import ChatOllamaModel
    import asyncio

    llm = ChatOllamaModel.chatOllama()
    query_data_service = QueryDataService()
    chat_history = "请帮我查询信号AUTOTEST1在昨天的持仓量数据"
    query ="""{'columns': 'trade_date, BOOK_NAME, CURRENT_HOLDING_QTY', 'filter_condition': 'BOOK_NAME = '自动化book17'", 'table': 'ficc_position', 
    'time_range':'trade_date BETWEEN 20240924 AND 20240925', 'userId': 'xtrader8888'}"""
    asyncio.run(query_data_service.query_data_from_sql_by_tools(query, llm))