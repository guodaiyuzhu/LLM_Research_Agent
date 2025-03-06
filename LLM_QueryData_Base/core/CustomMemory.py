from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.prompt import PromptTemplate

from LLM_RAG_Master.Utils.DB_Tools import DB
from langchain_core.messages import HumanMessage, AIMessage


class CustomMemory:
    @staticmethod
    def getConversationBufferMemory(**kwargs):
        return ConversationBufferMemory(**kwargs)

    @staticmethod
    def get_llm_chain(llm: BaseChatModel, memory: ConversationBufferMemory):
        template = "{chat_history}\n\nHuman: {input}"
        prompt = PromptTemplate(template=template, input_variables=["input"],
                                partial_variables={"chat_history": str(memory.chat_memory)}
                                )
        return prompt | llm

    @staticmethod
    def save_chat_history(data_json=None,
                          memory_json=None,
                          response_ai=None,
                          sql_info=None,
                          knowledge=None,
                          session_time=None):
        # data = {'user_id': '', 'session_id': '', 'contentType': '101', 'contentRef': ''}
        # session_id = data_json['sessionId']
        # user_id = data_json['user_id']
        # if session_id in memory_json:
        #     session_memory = memory_json[session_id]
        #     session_memory.sort(key=lambda x: x['history_id'])
        #     history_id_max = session_memory[-1]['history_id']
        # else:
        #     memory_json[session_id] = []
        #     history_id_max = 0
        # values = []
        # history_id_max += 1
        # chat_history = {
        #     'user_id': user_id,
        #     'session_id': session_id,
        #     'history_id': history_id_max,
        #     'query': data_json['contentRef'],
        #     'knowledge': knowledge,
        #     'response': response_ai,
        #     'sql_info': sql_info,
        #     'feedback': '',
        #     'eval_rag': None,
        #     'chat_date': session_time['chat_date'],
        #     'chat_start_time': session_time['chat_start_time'],
        #     'chat_end_time': session_time['chat_end_time']
        # }
        # memory_json[session_id].append(chat_history)
        # value = (chat_history['user_id'],
        #     chat_history['session_id'],
        #     chat_history['history_id'],
        #     chat_history['query'],
        #     chat_history['knowledge'],
        #     chat_history['response'],
        #     chat_history['sql_info'],
        #     chat_history['feedback'],
        #     chat_history['eval_reg'],
        #     chat_history['chat_date'],
        #     chat_history['chat_start_time'],
        #     chat_history['chat_end_time'])
        # values.append(value)
        #
        # sql_insert = {
        #     "insert into LLM_llm_session_history (user_id, session_id, history_id, name, memory, chat_date, chat_start_time, chat_end_time) values (%s, %s, %s, %s, %s, %s, %s, %s)"
        #     % (values[0][0], values[0][1], values[0][2], values[0][3], values[0][4], values[0][5], values[0][6], values[0][7])
        # }
        # sql_insert = (
        #     "insert into ficc_dwapp.LLM_llm_session_history"
        #     "(user_id, session_id, query, knowledge, response, sql_info, feedback, eval_rag, chat_date, chat_start_time, chat_end_time)"
        #     " values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
        # DB.insert_data_batch(sql=sql_insert, data_list=values, db_str="gp")
        return memory_json

    @staticmethod
    def get_memory_with_session_id(data_json, memory_json, history_duration):
        memory = CustomMemory.getConversationBufferMemory()
        session_id = data_json["sessionId"]
        if memory_json and session_id in memory_json:
            session_memory = memory_json[session_id]
            session_memory.sort(key=lambda x: x['history_id'])
            session_memory = session_memory[-history_duration:]
            for session_history in session_memory:
                role_name = session_history['role_name']
                if 'HUMAN' == role_name:
                    memory.chat_memory.add_message(HumanMessage(content=session_history['memory_history']))
                if 'AI' == role_name:
                    memory.chat_memory.add_message(AIMessage(content=session_history['memory_history']))
        return memory

    @staticmethod
    def get_memory_all():
        sql_search_history = ("select user_id, session_id, history_id, query, knowledge, response, sql_info, feedback, eval_rag, chat_date," 
                              "chat_start_time, chat_end_time from ficc_dwapp.LLM_llm_session_history t order by t.user_id asc, t.session_id asc, t.history_id asc")
        memory_json = {}
        # chat_history = DB.search_result(sql=sql_search_history, db_str='gp')
        # for i in chat_history:
        #     user_id,session_id,history_id,query,knowledge,response,sql_info,feedback,eval_rag,chat_date,chat_start_time,chat_end_time = 1
        #     if session_id not in memory_json:
        #         memory_json[session_id] = []
        #     memory_json[session_id].append({
        #         'user_id': user_id,
        #         'session_id': session_id,
        #         'history_id': history_id,
        #         'query': query,
        #         'knowledge': knowledge,
        #         'response': response,
        #         'sql_info': sql_info,
        #         'feedback': feedback,
        #         'eval_rag': eval_rag,
        #         'chat_date': chat_date,
        #         'chat_start_time': chat_start_time,
        #         'chat_end_time': chat_end_time
        #     })
        memory_json = [
            {
                    'user_id': '123',
                    'session_id': '123',
                    'history_id': '123',
                    'query': 'hello world',
                    'knowledge': 'basic knowledge',
                    'response': 'hello basic knowledge',
                    'sql_info': 'select basic knowledge from knowledge_database limit 10',
                    'feedback': 'full knowledge',
                    'eval_rag': '',
                    'chat_date': '20240801',
                    'chat_start_time': '20240801 172040',
                    'chat_end_time': '20240801 172060'
                }
        ]

        return memory_json

if __name__=='__main__':
    query = '请查询Autoriskbook004账号20240814的交易总金额'
    knowledge = 'bbbbb'
    response_ai = '{"total": 1}'
    data_json = {"userId":"xk","sessionId":"default","contentType":"101","contentRef":query}
    session_time = {"chat_date":"20240815","chat_start_time":"20240815 174122","chat_end_time":"20240815 175153"}
    sql_info="""SELECT SUM(trade_amt) FROM ficc_dwods.ficc_trade WHERE book_name = 'Autoriskbook004' AND trade_date='20240812' AND 1=1"""
    query_human = HumanMessage(content=query)
    response_ai = AIMessage(content=response_ai)
    memory_json = CustomMemory.get_memory_all()
    # memory_json = CustomMemory.save_chat_history(data_json, memory_json, response_ai, sql_info, knowledge, session_time)
    print(memory_json)
