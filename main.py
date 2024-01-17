import dgl
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import time
import os

from model import HGT


# 超参数
n_inp = 256     # 输入特征维度
n_hid = 128     # 隐藏层维度
n_epoch = 100    # 训练轮数
max_lr = 1e-3   # 最大学习率
clip = 1.0      # 梯度裁剪
n_layers = 1    # 层数
n_heads = 4     # 多头注意力
n_recommend = 3 # 推荐数量
n_classes = 13  # 类别数量
min_similarity = 0.4    # 最小相似度阈值
proportion = 0.8        # 训练集比例
data_origin = 2         # 1: 示例数据 2: 数据库数据

# 全局变量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
node_dict = {'person': 0}
edge_dict = {}
out_key = 'person'
train_idx = None
val_idx = None
labels = None
criterion = nn.BCEWithLogitsLoss()  # 多分类
data_path = "./data/example/" if data_origin == 1 else "./data/database/"
save_path = "./model/"


# 模型参数个数
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def create_hetero_graph():
    # 读取三元组文件
    with open(data_path + 'kg_final.txt', 'r', encoding='utf-8') as file:
        triples = [list(map(int, line.strip().split())) for line in file]

    # 读取关系映射文件
    relation_mapping = {}
    with open(data_path + 'relation_list.txt', 'r', encoding='utf-8') as file:
        for line in file:
            rel, rel_id = line.strip().split()
            edge_dict[rel] = int(rel_id)
            relation_mapping[int(rel_id)] = rel  # 将关系ID转换为整数

    # 区分三元组
    rel2person = {}
    for triple in triples:
        try:
            t1, t2 = rel2person[relation_mapping[triple[1]]]
        except:
            t1 = []
            t2 = []
        t1.append(triple[0]),  t2.append(triple[2])
        rel2person[relation_mapping[triple[1]]] = (t1, t2)

    # 建立三元组
    graph_data = {}
    for rel in rel2person:
        graph_data[('person', rel, 'person')] = rel2person[rel]

    # 创建异构图
    g = dgl.heterograph(graph_data)

    # 添加节点特征（随机初始化）
    num_nodes = g.num_nodes('person')
    node_features = torch.randn(num_nodes, n_inp)  # 64维的随机初始化特征
    g.nodes['person'].data['features'] = node_features

    return g.to(device)


def load_person():
    person_mapping = {}
    with open(data_path + 'person_list.txt', 'r', encoding='utf-8') as file:
        for line in file:
            p, pid, name = line.strip().split()
            person_mapping[int(pid)] = p
    return person_mapping


def load_train():
    global labels
    labels = []
    pids = []
    with open(data_path + 'train.txt', 'r', encoding='utf-8') as file:
        global n_classes
        n_classes = len(file.readline().strip().split()) - 1
        for line in file:
            res = line.strip().split()
            pid = int(res[0])
            label = [float(x) for x in res[1:]]
            labels.append(label)
            pids.append(pid)

    n = len(labels)
    n_train = int(n * proportion)   # 划分训练集
    original_idx = torch.arange(n).long()
    shuffle = np.random.permutation(original_idx)   # 随机打乱

    global train_idx, val_idx
    train_idx = torch.tensor(shuffle[:n_train]).long()
    val_idx = torch.tensor(shuffle[n_train:]).long()
    labels = torch.tensor(labels)


def load_model(g):
    model = HGT(
        g,
        node_dict,
        edge_dict,
        n_inp=n_inp,
        n_hid=n_hid,
        n_out=n_classes,
        n_layers=n_layers,
        n_heads=n_heads,
    ).to(device)
    return model


def train(model, G):
    best_val_acc = torch.tensor(0)
    best_model = None
    for epoch in np.arange(n_epoch) + 1:
        model.train()
        logits = model(G, out_key)
        loss = criterion(logits[train_idx], labels[train_idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            model.eval()
            logits = model(G, out_key)
            # 将logits转换为概率值
            probs = F.sigmoid(logits)
            # 预测的类别
            pred = (probs > 0.5).long().cpu()
            train_acc = (pred[train_idx] == labels[train_idx].long()).float().mean()
            val_acc = (pred[val_idx] == labels[val_idx].long()).float().mean()
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_model = model
            print(
                "Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f)"
                % (
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss.item(),
                    train_acc.item(),
                    val_acc.item(),
                    best_val_acc.item(),
                )
            )
    print("Best Val Acc %.4f" % (best_val_acc.item()))
    return best_model


def compute_cosine_similarity(model, graph):
    # 将模型设置为评估模式
    model.eval()

    # 在图上推断节点表示
    with torch.no_grad():
        # 将输入节点特征传递给模型，获取节点表示
        node_embeddings = model(graph, 'person')

    # 计算余弦相似度
    similarity_matrix = cosine_similarity(Tensor.cpu(node_embeddings))

    return similarity_matrix


def find_similar(similarity_matrix, n, k, person_mapping):
    similar_nodes = []
    for i in range(similarity_matrix.shape[0]):
        # 排除自身
        sim_scores = torch.tensor(similarity_matrix[i])
        sim_scores[i] = -1.0

        # 找到相似度最高的 n 个节点的索引
        top_k_indices = torch.topk(sim_scores, n).indices

        # 过滤保证相似度不小于 k 的节点
        selected_nodes = [person_mapping[idx.item()] for idx in top_k_indices if similarity_matrix[i, idx] >= k]

        similar_nodes.append(selected_nodes)

    res_list = []
    # 存储结果
    for i, nodes in enumerate(similar_nodes):
        res_list.append(f"Person {i}: Similar Persons {nodes}")
    return res_list


def save(model, res):
    # 将模型保存到save_path/当前时间 文件夹下
    save_dir_path = os.path.join(save_path, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    torch.save(model.state_dict(), os.path.join(save_dir_path, "model.pt"))
    print(f"模型已保存到 {save_dir_path}/model.pt")
    # 存储res到res.txt下
    with open(os.path.join(save_dir_path, "res.txt"), 'w', encoding='utf-8') as file:
        for line in res:
            file.write(line + '\n')
    print(f"结果已保存到 {save_dir_path}/res.txt")


if __name__ == '__main__':
    # 创建异构图数据集
    graph = create_hetero_graph()
    print(graph)

    # 加载训练集和模型
    load_train()
    model = load_model(graph)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, total_steps=n_epoch, max_lr=max_lr
    )

    # 训练
    print("Training MLP with #param: %d" % (get_n_params(model)))
    model = train(model, graph)

    # 计算余弦相似度矩阵
    matrix = compute_cosine_similarity(model, graph)
    print(matrix)

    # 推荐
    person_mapping = load_person()
    res_list = find_similar(matrix, n_recommend, min_similarity, person_mapping)

    # 保存结果
    save(model, res_list)
