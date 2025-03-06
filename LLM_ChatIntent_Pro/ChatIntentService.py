import os
import pandas as pd
from LLM_ChatIntent_Pro.detector import JointIntentSlotDetector
from transformers import BertTokenizer
from LLM_ChatIntent_Pro.model.models import JoinBert
from LLM_ChatIntent_Pro.data.datasets import IntentSlotDataset
from langchain_core.language_models import BaseChatModel
from LLM_RAG_Master.model.ChatOllama import ChatOllamaModel
import json

prompt_normal0 = """
你是一个分类器，请将用户提出的问题做分类，一类为Data,一类为Knowledge.
Data分类说明，用户提出类似查询数据，查询总金额，查询某某变化，请输出Data.
Knowledge分类说明，除了上述Data分类说明的场景，其他均为Knowledge. 

example:
Human: 我上个月的消费总额是多少？
AI: Data 

Human: 请查询上个月xx账号的持仓总金额
AI: Data 

Human: 如何优化我们的销售策略以提升利润？
AI: Knowledge 

Human: 我们的市场定位应该是什么样？
AI: Knowledge 

Human: {input}
AI:
"""

prompt_normal1 = """
你是一个金融行业智能问答机器人，拥有：ficc_portfolio_ana；
ficc_position；ficc_risk_analytics；ficc_trade；ficc_order。
你可以在以上提供的金融业务类型中选择回答。
下面提供几个样例供参考：
HUMAN：请查询资产池 A3 号资产的交易总金额
AI：[ficc_trade]
HUMAN：请查询资产池 A3 号资产与 202346883B 号资产的持仓总金额
AI：[ficc_position]
HUMAN：请查询持仓资产 3 天后的资产金额变化
AI：[ficc_position]
将 HUMAN 提出的问题做出选择回答：
HUMAN：{input}
AI：
"""
prompt_template2 = """
下面的描述是针对交易订单模块的，其中
持仓名称可以是交易持仓表，包含字段有当前持仓数量，多空方向等，后台名称：[ficc_position]。
交易流水名称可以是交易流水表，交易清单，或者交易，包含字段有全价，交易方向，净价（成交价后）
的差（成交价 - 净价），成交记录编号，成交价为 WSAP - 1 未成交，2 - 部分成交，3 - 全部成交，成交类型，成交数量。
交易金额，成交时间，到期日期，剩余到期天数，到期收益率，交易账户，后台名称：[ficc_trade]。
订单表名称可以是投资组合订单表，提单表，报表等，包含字段有金额，剩余期数，持仓时间，
交易金额，持仓收益，持仓时间，持仓金额，后台名称：[ficc_portfolio_ana]。
风险指标名称可以是风险指标，风险指标表，风险指标库等，包含字段有麦考利久期，应计利息，净价，CS01 1 - live，2 - eod，3 - sod，Fvbp，基点价值，到期日，类型，计算状态。
计算逻辑等，后台名称：[ficc_risk_analytics]。
订单表，后台名称：[ficc_order]。
在以下对话中识别出我们在语句中要查询的模块，输出：你要查询的是：[后台名称]。例如：
问题：最近一天的持仓数量是多少？
回答：你要查询的是：[ficc_position]
问题：今天的持仓价值是多少？
回答：你要查询的是：[ficc_position]
问题：你要查询的是：[ficc_portfolio_ana]
问题：今天，Book O2 最有一笔订单的净价是多少？
回答：你要查询的是：[ficc_order]
问题：{input}

"""
prompt_template3 = """
下面的描述是针对投资组合模块的，其中
持仓名称可以是交易持仓表，包含字段有当前持仓数量，多空方向等，后台名称：[ficc_position]。
交易流水名称可以是交易流水表，交易清单，或者交易，包含字段有全价，交易方向，净价（成交价后）
的差（成交价 - 净价），成交记录编号，成交价为 WSAP - 1 未成交，2 - 部分成交，3 - 全部成交，成交类型，成交数量。
交易金额，成交时间，到期日期，剩余到期天数，到期收益率，交易账户，后台名称：[ficc_trade]。
订单表名称可以是投资组合订单表，提单表，报表等，包含字段有金额，剩余期数，持仓时间，
交易金额，持仓收益，持仓时间，持仓金额，后台名称：[ficc_portfolio_ana]。
风险指标名称可以是风险指标，风险指标表，风险指标库等，包含字段有麦考利久期，应计利息，净价，CS01 1 - live，2 - eod，3 - sod，Fvbp，基点价值，到期日，类型，计算状态。
计算逻辑等，后台名称：[ficc_risk_analytics]。
订单表，后台名称：[ficc_order]。
在以下对话中识别出我们在语句中要查询的模块，输出：你要查询的是：[后台名称]。例如：
问题：最近一天的持仓数量是多少？
回答：你要查询的是：[ficc_position]
问题：今天的持仓价值是多少？
回答：你要查询的是：[ficc_position]
问题：你要查询的是：[ficc_portfolio_ana]
问题：今天，Book O2 最有一笔订单的净价是多少？
回答：你要查询的是：[ficc_order]
问题：{input}

"""

class ChatIntentService:
    def __init__(self, model_path):
        self.model_path = os.path.join(model_path, "bert-base-chinese/")
        self.data_path = os.path.join(model_path, "data/sample/test.json")
        self.tokenizer_path = os.path.join(model_path, "bert-base-chinese/")
        self.intent_path = os.path.join(model_path, "data/sample/intent_labels.txt")
        self.slot_path = os.path.join(model_path, "data/sample/slot_labels.txt")
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        self.dataset = IntentSlotDataset.load_from_path(
            data_path=self.data_path,
            intent_label_path=self.intent_path,
            slot_label_path=self.slot_path,
            tokenizer=self.tokenizer)
        self.model = JoinBert.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            slot_label_num=7,
            intent_label_num=3
        )
        self.detector = JointIntentSlotDetector(
            model=self.model,
            tokenizer=self.tokenizer,
            intent_dict=self.dataset.intent_label_dict,
            slot_dict=self.dataset.slot_label_dict
        )
        self.prompt_normal = prompt_normal0

    async def detect(self, query):
        intent_json = self.detector.detect(query)
        return intent_json

    async def detect_ollama(self, query, llm: BaseChatModel):
        prompt_tmp = self.prompt_normal.format(input=query)
        res = llm.invoke(prompt_tmp)
        print(res)
        return {'text': query, 'intent': res.content}

    def test_query_dict(self, query_dict, llm: BaseChatModel):
        res_list = []
        for query in query_dict.keys():
            intent_res = self.detect_ollama(query, llm)
            if " " not in intent_res:
                continue
            detected_ans = intent_res.split(" ")[-1].split("'")[0]
            res = (query, detected_ans, query_dict[query], "label")
            res_list.append(res)
        res_pd = pd.DataFrame(res_list, columns=["query", "detected_intent", "true_intent", "label"])
        return res_pd


def load_queries(file_name):
    list = []
    with open(file_name, 'r') as f:
        for line in f:
            list.append(line)
    return list


if __name__ == "__main__":
    import asyncio
    service = ChatIntentService(os.getcwd())
    query = "Hello, world!"
    llm = ChatOllamaModel.chatOllama()
    intent_json = asyncio.run(service.detect_ollama(query,llm=llm))
    print(intent_json)