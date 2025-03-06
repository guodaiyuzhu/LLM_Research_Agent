import datetime
import glob
import json
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt import ChatPromptTemplate, MessagePlaceholder
from FICC_RAG_Master.model.ChatOllama import ChatOllamaModel
from langchain_core.messages import SystemMessage

slot_and_inquiry_prompt_template="""
你是一个专门处理数据查询的智能机器人，参考注意事项和背景知识，遵循以下处理流程：

*背景知识*：
{topic_base_info}

*处理流程*
1. **判断查询意图**：
    - 若用户查询关于{topic_name}数据，则按{topic_name}数据背景知识问答。
    - 若用户查询内容与{topic_name}数据无关，请回答： response="你好，我擅长查询{topic_name}相关数据的机器人，其他数据请在相应主题问询，谢谢！"

2. **要素解析**：
    - 用户查询时，解析出四个要素： columns， table， time_range, filter_condition;
    - columns要素确认添加 trade_date 字段； time_range要素内容不为空， filter_condition要素内容必须又book_name;
    - 用JSON格式输出，例如： params={{"columns":"trade_date, pos_qty","table":"ficc_position","time_range":"trade_date=20240301","filter_conditon":"BOOK=大宗台"}}

3. **要素追问**：
    - 当某个要素解析完成后，依次对下一个要素进行追问，直到四个要素全部获取完成。
    - 当四个要素全部解析完成后，必须询问用户确认，并按照如下回复用户， ： response="您好， 您查询需求的关键要素如下，您可以修改也可直接回复确认：字段为： $columns; 表名为：$table; 时间为： $time_range", 筛选条件： $filter_condition;" ; 如果用户并未确认或者有修改，请在此追问和抽取要素。
    - 当收到用户确认回复后，请按照如下回复： response="好的，正在为您查询数据，请稍后..."

4. **最终输出**：
    - 你的回答放入response中，最终只能输出一个包含所有状态及对话内容的JSON, 不得输出其他格式，格式如下：
    {{
        "params": "$params", 
        "intent_state": "$intent_state",
        "inquiry_state": "$inquiry_state",
        "response":"$response"
    }}

*注意事项*
1. **时间time_range与筛选条件filter_condition**:
    - 当前系统时间为 {current_date};
    - 时间格式为yyyyMMdd, 例如"trade_date=20240818"
    - 筛选条件：用户问题中涉及到账户名，金融品种，交易所等等，格式化例如"book_name=大宗台"，"book_name=BOOK002","security_type=1","side=2";

2. **状态标识与要素确认**：
    - 在整个对话过程中，输出状态标识intent_state和inquiry_state, 分别标识意图状态和中间状态。
    - 如果用户回答与要素相关且要素缺失，intent_state=IN, inquiry_state=START。
    - 如果用户回答与要素无关，intent_state=OUT， inquiry_state=STOP。
    - 要素确认后，若用户未确认或修改，intent_state=IN, inquiry_state=START；用户确认后，intent_state=IN, inquiry_state=STOP。

"""
class ChatInquiryService:
    @staticmethod
    def load_base_info():
        all_base_infos = {}
        current_file = os.path.abspath(os.path.dirname(__file__))
        json_path = os.path.join(current_file, "../..//FICC_Rate_Master/config//**/*.json")
        for file_path in glob.glob(json_path, recursive=True):
            # current_config = load_scene_templates(file_path)
            with open(file_path, 'r', encoding='utf-8') as file:
                current_config = json.load(file)
            for key, value in current_config.items():
                if key not in all_base_infos:
                    all_base_infos[key] = value
        return all_base_infos

    def __init__(self):
        self.topic_name_dict = {"220001": "ficc_position", "220002": "ficc_trade", "220003": "ficc_port_pnl", "220004": "ficc_risk_analytics"}
        self.topic_base_infos=ChatInquiryService.load_base_info()
        self.llm = ChatOllamaModel.chatOllama()

    def get_name_by_topic(self, topic):
        return self.topic_name_dict.get(topic)

    def get_topic_base_info(self, topic):
        topic_name = self.get_name_by_topic(topic)
        return self.get_topic_base_info_by_name(topic_name)

    def get_topic_base_info_by_name(self, topic_name):
        topic_info = self.topic_base_infos.get(topic_name)
        return topic_info.get("base_info")

    def slot_and_inquiry(self, messages:list, topic):
        topic_name = self.get_name_by_topic(topic)
        topic_base_info = self.get_topic_base_info_by_name(topic_name)
        e = datetime.datetime.now()
        current_date = e.strftime("%Y-%m-%d")
        slot_and_inquiry_prompt = slot_and_inquiry_prompt_template.format(topic_name=topic_name,
                                                                          topic_base_info=topic_base_info,
                                                                          current_date=current_date)
        # print(slot_and_inquiry_prompt)
        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=slot_and_inquiry_prompt
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        messages_json = "messages"
        chain = chat_template | self.llm | StrOutputParser()
        response = chain.invoke(messages_json)
        return response

if __name__ == "__main__":
    service = ChatInquiryService()
    inquiry = service.slot_and_inquiry("查询持仓信息", "220001")
    print('='*30)
    print(inquiry)