import time
import gradio as gr
from DB_Tools import DB

class FeedBack:
    def __init__(self):
        self.db = DB()

    def feedback(self, chatbot, txt, flag, req: gr.Request):
        table = 'ficc_dwods.ficc_llm_evaluation'
        rows = ['trade_date', 'query', 'evaluation', 'favorite', 'client_ip']
        rows_total = ','.join(rows)
        values = ','.join(['%s' for i in range(0, len(rows), 1)])
        insert_sql = "INSERT INTO {s1} ({s2}) VALUES ({s3})".format(s1=table, s2=rows_total, s3=values)

        t = time.time() - 60 * 60 * 24 * 30
        time_string = time.strftime("%Y%m%d%H%M%S", time.localtime(t))
        client_ip = req.client.host

        if len(chatbot) > 1:
            chatbot_temp = [chatbot[i] for i in range(1, len(chatbot), 1)]
        else:
            chatbot_temp = []

        if flag == '0':
            if chatbot_temp == []:
                gr.Info('感谢您的反馈, 请关注后续版本更新.')
            else:
                data_list = [(time_string, chatbot_temp, None, txt, client_ip)]
                self.db.insert_data_batch(insert_sql, data_list)
                gr.Info('感谢您的反馈, 请关注后续版本更新.')
        else:
            if txt == '':
                gr.Warning('请输入内容后再提交')
                return gr.update(value='')
            else:
                data_list = [(time_string, chatbot_temp, txt, None, client_ip)]
                self.db.insert_data_batch(insert_sql, data_list)
                gr.Info('提交成功, 感谢您的反馈, 请关注后续版本更新.')
                return gr.update(value='')
