import json
import os
import pandas as pd
import pymysql

data_path = "./data/database"
df = pd.read_excel(os.path.join(data_path, "tw_kg_subgraph.xlsx"), )
with open("database.json", 'r') as file:
    database = json.load(file)

# 打印df统计信息
def df_info(df):
    print("info:")
    print(df.info())
    print("head:")
    print(df.head())
    # r_type_counts = df['r_type'].value_counts()
    # print(r_type_counts)


def filter_df(df):
    return df[df['b_type'] == '人物']


def get_relation_mapping(df):
    edge_dict = {}
    relation_mapping = {}
    r_type = df['r_type'].unique()
    for i in range(len(r_type)):
        edge_dict[r_type[i]] = i
        relation_mapping[i] = r_type[i]
    return edge_dict, relation_mapping


def get_person_mapping(df):
    person_mapping = {}
    person_id_mapping = {}
    id2name = {}
    person = pd.Series(pd.concat([df['a_id'], df['b_id']])).unique()
    for i in range(len(person)):
        person_mapping[person[i]] = i
        person_id_mapping[i] = person[i]
    # 获取 a_id 到 a_name 的映射
    a_mapping = df.groupby('a_id')['a_name'].unique().apply(lambda x: x[0]).to_dict()
    # 获取 b_id 到 b_name 的映射
    b_mapping = df.groupby('b_id')['b_name'].unique().apply(lambda x: x[0]).to_dict()
    # 将两个映射合并为一个字典
    id2name = {**a_mapping, **b_mapping}
    return person_mapping, person_id_mapping,id2name


def get_kg_final(df, person_mapping, edge_dict):
    df = df[['a_id', 'r_type', 'b_id']]
    df.loc[:, 'a_id'] = df['a_id'].map(person_mapping)
    df.loc[:, 'b_id'] = df['b_id'].map(person_mapping)
    df.loc[:, 'r_type'] = df['r_type'].map(edge_dict)
    print(df)
    return df


def get_type_mapping(person_list):
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
        return
    try:
        with connection.cursor() as cursor:
            # 执行查询
            sql = "SELECT id, person_id, type, subtype FROM t_biz_person_tags WHERE person_id IN (%s)" % ','.join(['%s'] * len(person_list))
            # print(person_list)
            cursor.execute(sql, person_list)

            results = cursor.fetchall()
            df = pd.DataFrame(results)
            subtype = df['subtype'].unique()
            type_mapping = {}
            for i in range(len(subtype)):
                type_mapping[subtype[i]] = i


    finally:
        # 关闭数据库连接
        connection.close()
    return type_mapping, df


def get_train(df, person_mapping):
    df.loc[:, 'pid'] = df['person_id'].map(person_mapping)
    # 使用 pivot_table 透视数据
    pivot_df = df.pivot_table(index='pid', columns='subtype', values='type', aggfunc='count', fill_value=0)

    # 重置索引并设置列名
    pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)

    # 将 NaN 替换为 0
    pivot_df = pivot_df.fillna(0)
    return pivot_df


def save(relation_mapping, person_mapping, id2name, kg_final, type_mapping, pivot_df):
    with open(os.path.join(data_path, 'relation_list.txt'), 'w', encoding='utf-8') as file:
        for k, v in relation_mapping.items():
            file.write(str(k) + '\t' + str(v) + '\n')
    with open(os.path.join(data_path, 'person_list.txt'), 'w', encoding='utf-8') as file:
        for k in person_mapping.keys():
            file.write(str(k) + '\t' + str(person_mapping[k]) + '\t' + id2name[k] + '\n')
    with open(os.path.join(data_path, 'kg_final.txt'), 'w', encoding='utf-8') as file:
        for index, row in kg_final.iterrows():
            triple = f"{row['a_id']}\t{row['r_type']}\t{row['b_id']}"
            file.write(f"{triple}\n")
    with open(os.path.join(data_path, 'label_list.txt'), 'w', encoding='utf-8') as file:
        for k, v in type_mapping.items():
            file.write(str(k) + '\t' + str(v) + '\n')
    pivot_df.to_csv(os.path.join(data_path, 'train.txt'), sep='\t', index=False, header=False, encoding='utf-8')


if __name__ == '__main__':
    df = filter_df(df)
    df_info(df)
    e, r = get_relation_mapping(df)
    p, pi, name = get_person_mapping(df)
    df_kg = get_kg_final(df, p, e)
    type_mapping, df_label = get_type_mapping(list(p.keys()))
    pivot_df = get_train(df_label, p)
    save(e, p, name, df_kg, type_mapping, pivot_df)
