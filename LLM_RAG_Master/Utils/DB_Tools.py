import psycopg2
import base64
import cx_Oracle
from LLM_RAG_Master.config.config import *

class DB:
    @staticmethod
    def get_gp_conn(gp_conn=DEFAULT_GP_CONNECTION):
        """获取GP连接"""
        # conn = psycopg2.connect(
        #     host=gp_conn["host"],
        #     port=gp_conn["port"],
        #     database=gp_conn["database"],
        #     user=gp_conn["user"],
        #     # password=str(base64.b64decode(bytes(gp_conn["password"], 'utf-8')), 'utf-8'),
        #     password = "123456",
        # )
        conn = None
        return conn

    @staticmethod
    def get_oracle_conn(ora_conn=DEFAULT_ORACLE_CONNECTION):
        """获取Oracle连接"""
        host = ora_conn["host"]
        port = ora_conn["port"]
        database = ora_conn["database"]
        user = ora_conn["user"]
        password = str(base64.b64decode(bytes(ora_conn["password"], 'utf-8')), 'utf-8')
        conn = cx_Oracle.connect(f'{user}/{password}@{host}:{port}/{database}')
        return conn

    @staticmethod
    def insert_data_batch(sql, data_list: list[tuple], db_str='ora'):
        if db_str == 'ora':
            conn = DB.get_oracle_conn()
        elif db_str == 'gp':
            conn = DB.get_gp_conn()
        else:
            conn = DB.get_oracle_conn()

        cursor = conn.cursor()
        try:
            # 批量执行SQL语句，注意 data_list 格式 list[tuple()]
            cursor.executemany(sql, data_list)
            # 提交事务
            conn.commit()
        finally:
            # 关闭游标和连接
            cursor.close()
            conn.close()

    @staticmethod
    def search_result(sql, db_str='ora'):
        if db_str == 'ora':
            conn = DB.get_oracle_conn()
        elif db_str == 'gp':
            conn = DB.get_gp_conn()
        else:
            conn = DB.get_oracle_conn()

        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
        finally:
            cursor.close()
            conn.close()
        return result
