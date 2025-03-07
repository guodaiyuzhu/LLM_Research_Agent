import re
import json
import requests
from requests import request
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool

def query_from_mysql(sql, env):
    import pymysql

    if env == 'AT':
        mysql_conn_json = {
            'host': '',
            'port': 3307,
            'user': '',
            'password': '',
        }
    elif env == 'SIT':
        mysql_conn_json = {
            'host': '',
            'port': 3306,
            'user': '',
            'password': '',
        }
    else:
        mysql_conn_json = {
            'host': '',
            'port': 3306,
            'user': '',
            'password': '',
        }

    db = pymysql.connect(
        host=mysql_conn_json["host"],
        port=mysql_conn_json["port"],
        user=mysql_conn_json["user"],
        password=mysql_conn_json["password"]
    )

    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    cursor.execute(sql)
    data = cursor.fetchall()

    # 关闭数据库连接
    db.close()

    return data

def extract_sql_info(sql):
    # 匹配Select语句中的字段信息
    select_pattern = r'select\s+(.*?)\s+from'
    select_match = re.search(select_pattern, sql, re.IGNORECASE)
    if select_match:
        select_info = select_match.group(1).strip()
    else:
        select_info = None

    # 匹配From语句中的表信息
    from_pattern = r'from\s+(.*?)\s+where'
    from_match = re.search(from_pattern, sql, re.IGNORECASE)
    if from_match:
        from_info = from_match.group(1).strip()
    else:
        from_info = None

    # 匹配Where语句中的条件信息
    where_pattern = r'where\s+(.*)$'
    where_match = re.search(where_pattern, sql, re.IGNORECASE)
    if where_match:
        where_info = where_match.group(1).strip()
    else:
        where_info = None

    return select_info, from_info, where_info

class DataAPI(BaseModel):
    sql: str=Field(description="sql scripts")

@tool("query data from sql", args_schema=DataAPI)
def get_data_from_data_api(sql:str):
    """
    get data from sql
    Args:
        sql: sql script
    """
    print(f'sql:{sql}')
    columns_info, table_info, condition_info=extract_sql_info(sql)
    context={
        "userId":"xtrader8888",
        "column":columns_info,
        "table":table_info,
        "condition":condition_info
    }
    url = "http://168.64.9.220:8080/fanedataapi/api/ficc/ai/dataQuery"
    response_data=requests.post(url=url,json=context)
    res=response_data.text
    return res, context 

class ExtractSQL(BaseModel):
    table_name: str=Field(description="table names")
    columns: str|list=Field(description="all columns")
    condition: str=Field(description="condition")
    logic: str=Field(description="logic")

@tool(args_schema=ExtractSQL, return_direct=True)
def extract_elements_from_sql(table_name,columns,condition,logic):
    """
    Extracting elements from sql
    Args
        table_name : table name with database or schema
        columns : table columns from sql
        condition : conditions from sql
        logic : logic from sql, such as "SUM", "AVG"...
    """
    print(f'table_name:{table_name},columns:{columns},condition:{condition},logic:{logic}')
    return {"result":"success"}


class ExtractUserName(BaseModel):
    user_name: str=Field(description="user names")
    locations: str|list=Field(description="all columns")

@tool(args_schema=ExtractUserName, return_direct=True)
def extract_username_from_inputs(user_name:str,locations:str|list):
    """
    Extracting user names
    Args
        user_name : user name in prompts
        locations : user locations
    """
    print(f'the response is user_name:{user_name},locations:{locations}')
    return {"result":"success"}
