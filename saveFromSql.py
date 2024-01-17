import json
import os
import pymysql
import pandas as pd
from prettytable import PrettyTable
import openpyxl


data_path = "./data/database"
batch_size = 500  # 每次读取的记录数
numbers = 100000  # 读取的记录总数, -1表示全部读取
graph = 1  # 1表示读取subgraph，0表示读取全部图

with open("database.json", 'r') as file:
    database = json.load(file)

try:
    connection = pymysql.connect(
        host=database['host'],
        user=database['user'],
        password=database['password'],
        port=database['port'],
        db=database['db'],
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    print("数据库连接成功")
except:
    print("数据库连接失败")
    exit(0)


try:
    # 创建游标对象
    with connection.cursor() as cursor:
        if numbers == -1:
            sql = "SELECT * FROM tw_kg_subgraph"
        else:
            sql = "SELECT * FROM tw_kg_subgraph LIMIT %d" % numbers
        cursor.execute(sql)

        # 获取查询结果
        results = cursor.fetchall()
        size = results.__len__()

        # 如果结果集不为空
        if results:
            # 使用第一行的值作为列名
            table = PrettyTable(field_names=results[0].keys())

            # 添加数据到表格
            for row in results:
                table.add_row(row.values())

            df_list = []
            # 转换为 pandas DataFrame
            for i in range(0, len(results), batch_size):
                df_list.append(pd.DataFrame(list(results[i:i+batch_size])))
                print(f'已读取 {i+batch_size} 条记录，共 {size} 条记录。')

            df = pd.concat(df_list, ignore_index=True)

            # 保存为 Excel 文件
            excel_filename = os.path.join(data_path, "tw_kg_subgraph.xlsx")
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            df.to_excel(excel_filename, index=False)
            print(f"表格已保存为 {excel_filename}")
        else:
            print("结果集为空")

finally:
    # 关闭数据库连接
    connection.close()
