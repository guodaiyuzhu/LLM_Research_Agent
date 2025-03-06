from langchain.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from FICC_RAG_Master.model.ChatOllama import ChatOllamaModel

chat_intent_template = """
你是一个意图识别专家，判断用户意图，共分三类：Data，Knowledge，Chat.
其中三种意图的解释如下：
Data，此类问题需要从特定数据库或系统中查询数据的问题。这些问题通常涉及具体的数据、记录或报表内容。
Knowledge，此类问题包括询问某个实体的定义、解释、操作步骤等的问题。
Chat，此类问题包括日常对话、简单的话题讨论、不涉及特定信息查询的问题。例如，问候，个人心情，闲聊等。

注意：
1. 无法判断用户意图时，统一归类为Chat.
2. 只输出意图标签：Data，Knowledge，Chat.
3. 必须格式化为：{"intent":"$intent"}
"""


class ChatIntentService:
    def __init__(self):
        # print("ChatIntentService init success")
        self.llm = ChatOllamaModel.chatOllama()
        self.chat_intent_template = chat_intent_template

    async def intent(self, query: str):
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=self.chat_intent_template
                ),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        chain = prompt | self.llm
        messages_json = {"messages": [HumanMessage(content=query)]}
        res = chain.invoke(messages_json)
        res = json.loads(res.content)["intent"]
        # print(res)
        return res


if __name__ == "__main__":
    import asyncio
    chat_intent_service = ChatIntentService()
    async def main(query):
        res = await chat_intent_service.intent(query)
        print(res)
   
    asyncio.run(main("写一篇文章，字数300个，主题内容：SpaceX"))
    asyncio.run(main("我要去北京"))
    asyncio.run(main("风险管理人员的主要工作内容是什么？"))