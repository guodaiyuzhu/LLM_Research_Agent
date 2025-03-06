import json  
from datetime import datetime  
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage  
from LLM_ChatIntent_Pro.ChatIntentService import ChatIntentService  
from LLM_QueryData_Base.core.QueryDataService import QueryDataService  
from LLM_QueryData_Base.core.CustomMemory import CustomMemory  
  
  
class ControllerService:  

    def __init__(self, chat_intent_model_path):  
        self.query_data_service = QueryDataService()   
        self.intent_model_path = chat_intent_model_path  
        self.chat_intent = ChatIntentService(model_path=self.intent_model_path)  
        self.memory_list = CustomMemory.get_memory_all()
        self.history_duration = 6  
        self.session_time = {"chat_date": "", "chat_start_time": "", "chat_end_time": ""}  

    async def predict(self, query, llm: BaseChatModel):  
        symbol_start = {'type': 'symbol', 'value': '<message_start>'}  
        symbol_end = {'type': 'symbol', 'value': '<message_end>'}  
        self.session_time["chat_date"] = datetime.now().strftime("%Y%m%d")  
        self.session_time["chat_start_time"] = datetime.now().strftime("%Y%m%d %H%M%S")  
        yield symbol_start  
        try:  
            query = query.replace('\n','\\n')  
            data_json = json.loads(query)  
            if all(key in data_json for key in ["sessionId", "contentType", "contentRef"]):
                memory = CustomMemory.get_memory_with_session_id(data_json, self.memory_list, self.history_duration)
                data_content = data_json["contentRef"]
                if data_content != '':  
                    # intent_json = await self.chat_intent.detect（data_content)  
                    intent_json = await (self.chat_intent.detect_ollama(data_content, llm))
                    print(f'intent_json: {intent_json}')
                    # 判断问答场景  
                    if intent_json['intent'] == 'Data':  
                        """数据查询"""  
                        response_tools, args = await (self.query_data_service.query_data_from_sql_by_tools(data_content, llm))
                        # response = llm.invoke(query)  
                        try:  
                            response_tools_json = json.loads(response_tools)  
                            res_data_api = response_tools_json['list']
                        except json.JSONDecodeError:  
                            res_data_api = response_tools  

                        memory_temp = memory
                        memory_temp.chat_memory.add_messages([
                            HumanMessage(content=data_content),
                            AIMessage(content=res_data_api)
                        ])
                        # print(memory_temp.chat_memory)  # 开启历史消息  
                        
                        llm_chain = CustomMemory.get_llm_chain(llm=llm, memory=memory_temp)  # 流式回答  
                        
                        yield {  
                            'type': 'symbol',  
                            'value': '<text_start>'  
                        }  
                        # 存储流式回答的内容到memory  
                        response_ai = ''  
                        
                        async for res in llm_chain.astream({  
                            'input': data_content  
                        }):  
                            # print(f"response :{response.content}")  # 注意：这里可能是 res.content 而不是 response.content  
                            response_ai += res.content  
                            yield {  
                                'type': 'text',  
                                'value': res.content  
                            }  
                            
                        yield {  
                            'type': 'symbol',  
                            'value': '<text_end>'  
                        }  
                            
                        query_human = HumanMessage(content=data_content)  
                        response_ai = AIMessage(content=response_ai)  
                        
                        self.session_time["chat_end_time"] = datetime.now().strftime("%Y%m%d %H%M%S")  
                        self.memory_list = CustomMemory.save_chat_history(
                            data_json,
                            self.memory_list,
                            [query_human, response_ai],
                            self.session_time
                        )
                        # print(memory.chat_memory)  
                        x = json.loads(response_tools)['total'] 
                        if x > 10: 
                            yield {'type': 'symbol', 'value': '<link_start>'}  
                            yield {'type': 'link', 'value': json.dumps(args)}  
                            yield {'type': 'symbol', 'value': '<link_end>'}  
                    elif intent_json['intent'] == 'Knowledge':
                        llm_chain = CustomMemory.get_llm_chain(llm=llm, memory=memory)
                        yield symbol_start
                        yield {'type': 'symbol', 'value': '<text_start>'}
                        response_ai = ''
                        async for res in llm_chain.astream(data_content):   
                            print(f"response :{res.content}")  
                            response_ai += res.content  
                            yield {'type': 'text', 'value': res.content}  
                        yield {'type': 'symbol', 'value': '<text_end>'}   
                        query_human = HumanMessage(content=data_content)  
                        response_ai = AIMessage(content=response_ai)  
                        self.session_time["chat_end_time"] = datetime.now().strftime("%Y%m%d %H%M%S")  
                        self.memory_list = CustomMemory.save_chat_history(
                                data_json,  # 注意：这里可能需要传递更多的参数，但根据文本只能确定 data_json
                                self.memory_list,
                                [query_human, response_ai],
                                self.session_time)
                              
                    else:  
                        normal_list = [  
                            {'type': 'symbol', 'value': '<text_start>'},  
                            {'type': 'text', 'value': '请输入文本后再提交'},  
                            {'type': 'symbol', 'value': '<text_end>'},
                            symbol_end
                        ]   
                        for i in normal_list:
                            yield i
                else:
                    yield {'type':'text','value':"Error: The data format is incorrect."}
        except json.JSONDecodeError:
            yield {'type':'text','value':"Error: The data format is not json."}
        except Exception as e:
            print(e)
            yield {'type': 'text','value':f"Error: {e}"}
        yield symbol_end

if __name__=="__main__":
    from LLM_RAG_Master.model.ChatOllama import ChatOllamaModel
    import asyncio
    import os

    async def main(query_content='hello'):
        current_directory = "/home/jerryxu/Downloads/LLM"
        chat_intent_model_path = f"{current_directory}/LLM_ChatIntent_Pro/"
        llm = ChatOllamaModel.chatOllama(model='qwen2:7b')
        # memory = CustomMemory.getConversationBufferMemory(memory_key="chat_history")
        query = '{\"sessionId":"yz","contentType":"str","contentRef": "'+query_content+'"}'
        print(query)
        query = query.replace('\n', '\\n')
        data_json = json.loads(query)
        print(data_json)
        controller= ControllerService(chat_intent_model_path)
        async for item in controller.predict(query,llm):
            print(item)
        
    # asyncio.run(main(query_content='请查询BOOKTrader03今天的持仓金额'))
    asyncio.run(main(query_content='今天天气真好呀'))