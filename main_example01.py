import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import random

# 超参数
NUM_NODES = 300
FEATURE_DIM = 10
NUM_CLASSES = 7
NUM_EDGES = 60
NUM_CLIENTS = 3
EPOCHS = 10
ROUNDS = 5

# 1. 生成节点特征、标签、边连接
np.random.seed(0)
node_features = np.random.randint(2, size=(NUM_NODES, FEATURE_DIM))
node_labels = np.random.randint(NUM_CLASSES, size=(NUM_NODES, 1))
nodes = np.hstack((np.arange(NUM_NODES).reshape(-1, 1), node_features, node_labels))

# 生成无向边
edges = np.array([(random.randint(0, NUM_NODES - 1), random.randint(0, NUM_NODES - 1)) for _ in range(NUM_EDGES)])

# 将图划分为三个子图，每个子图100个节点
subgraph_size = NUM_NODES // NUM_CLIENTS
subgraphs = []
for i in range(NUM_CLIENTS):
    start = i * subgraph_size
    end = start + subgraph_size
    subgraph_nodes = nodes[start:end]
    subgraph_edges = [(u, v) for u, v in edges if start <= u < end and start <= v < end]
    subgraphs.append((subgraph_nodes, subgraph_edges))

# 2. 定义简单的神经网络模型
class SimpleNodeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleNodeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 3. 客户端本地训练
def client_update(client_model, data, epochs=EPOCHS):
    optimizer = optim.Adam(client_model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(epochs):
        client_model.train()
        optimizer.zero_grad()
        features, labels = torch.tensor(data[:, 1:-1], dtype=torch.float32), torch.tensor(data[:, -1], dtype=torch.long)
        output = client_model(features)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
    return client_model.state_dict()

# 4. 联邦平均算法聚合模型参数
def federated_averaging(global_model, client_states):
    global_state = global_model.state_dict()
    for key in global_state.keys():
        global_state[key] = torch.stack([client_state[key].float() for client_state in client_states]).mean(dim=0)
    global_model.load_state_dict(global_state)

# 5. 联邦学习主过程
def federated_training():
    global_model = SimpleNodeClassifier(input_dim=FEATURE_DIM, hidden_dim=16, num_classes=NUM_CLASSES)
    client_models = [SimpleNodeClassifier(input_dim=FEATURE_DIM, hidden_dim=16, num_classes=NUM_CLASSES) for _ in range(NUM_CLIENTS)]

    for round in range(ROUNDS):
        client_states = []
        for i in range(NUM_CLIENTS):
            # 每个客户端加载全局模型的参数
            client_models[i].load_state_dict(global_model.state_dict())
            # 在客户端本地数据上训练模型
            client_state = client_update(client_models[i], subgraphs[i][0], epochs=EPOCHS)
            client_states.append(client_state)

        # 聚合客户端模型参数
        federated_averaging(global_model, client_states)
        print(f"Round {round + 1} completed.")

    # 全局模型评估
    evaluate_global_model(global_model)

# 6. 测试全局模型
def evaluate_global_model(global_model):
    global_model.eval()
    correct = 0
    total = 0
    for nodes, _ in subgraphs:
        features, labels = torch.tensor(nodes[:, 1:-1], dtype=torch.float32), torch.tensor(nodes[:, -1], dtype=torch.long)
        output = global_model(features)
        pred = output.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    print(f"Global Model Accuracy: {accuracy * 100:.2f}%")

# 执行联邦学习
federated_training()
