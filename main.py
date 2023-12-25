import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor

from model import HGT


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_inp = 256
n_hid = 128
n_epoch = 200
max_lr = 1e-3
clip = 1.0
node_dict = {'person': 0}
edge_dict = {}
n_recommend = 2
min_similarity = 0.0


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
    with open('kg_final.txt', 'r') as file:
        triples = [list(map(int, line.strip().split())) for line in file]

    # 读取关系映射文件
    relation_mapping = {}
    with open('relation_list.txt', 'r') as file:
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


def load_model(g):
    model = HGT(
        g,
        node_dict,
        edge_dict,
        n_inp=n_inp,
        n_hid=n_hid,
        n_out=5,
        n_layers=1,
        n_heads=4,
    ).to(device)
    return model


def train(model, G):
    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)
    for epoch in np.arange(n_epoch) + 1:
        model.train()
        logits = model(G, "paper")
        # The loss is computed only for labeled nodes.
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
        if epoch % 5 == 0:
            model.eval()
            logits = model(G, "paper")
            pred = logits.argmax(1).cpu()
            train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
            val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
            test_acc = (pred[test_idx] == labels[test_idx]).float().mean()
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
            print(
                "Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)"
                % (
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss.item(),
                    train_acc.item(),
                    val_acc.item(),
                    best_val_acc.item(),
                    test_acc.item(),
                    best_test_acc.item(),
                )
            )


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


def find_similar(similarity_matrix, n, k):
    similar_nodes = []
    for i in range(similarity_matrix.shape[0]):
        # 排除自身
        sim_scores = torch.tensor(similarity_matrix[i])
        sim_scores[i] = -1.0

        # 找到相似度最高的 k 个节点的索引
        top_k_indices = torch.topk(sim_scores, n).indices

        # 过滤保证相似度不小于 t 的节点
        selected_nodes = [idx.item() + 1 for idx in top_k_indices if similarity_matrix[i, idx] >= k]

        similar_nodes.append(selected_nodes)

    # 输出结果
    for i, nodes in enumerate(similar_nodes):
        print(f"Person {i + 1}: Similar Nodes {nodes}")


if __name__ == '__main__':
    # 创建异构图数据集
    graph = create_hetero_graph()
    print(graph)

    model = load_model(graph)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, total_steps=n_epoch, max_lr=max_lr
    )

    # 训练
    # print("Training MLP with #param: %d" % (get_n_params(model)))
    # train(model, graph)

    # 计算余弦相似度矩阵
    matrix = compute_cosine_similarity(model, graph)
    print(matrix)

    # 推荐
    find_similar(matrix, n_recommend, min_similarity)
