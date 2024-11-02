import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from transformers.models.bert.modeling_bert import BertPreTrainedModel

from EvaluateAcc import EvaluateAcc
from MethodGraphBert import MethodGraphBert
import config
from DatasetLoader import DatasetLoader
from MethodBertComp import GraphBertConfig

BertLayerNorm = torch.nn.LayerNorm

dataset_name = config.dataset_name

np.random.seed(1)
torch.manual_seed(1)

nclass = 7
nfeature = 1433





lr = config.lr
k = config.k_hop
max_epoch = config.max_epoch

x_size = nfeature
hidden_size = intermediate_size = 32
num_attention_heads = config.attention_heads
num_hidden_layers = config.hidden_layers
y_size = nclass

residual_type = 'graph_raw'

def load_data(dataset_name):
    data_obj = DatasetLoader()
    data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
    data_obj.dataset_name = dataset_name
    data_obj.k = config.k_hop
    data_obj.load_all_tag = True
    return data_obj.load()


def Classifier(dataset_name):
    print(dataset_name, end="\t")
    bert_config = GraphBertConfig(residual_type=residual_type, k=k, x_size=nfeature, y_size=y_size,
                                  hidden_size=hidden_size,
                                  intermediate_size=intermediate_size, num_attention_heads=num_attention_heads,
                                  num_hidden_layers=num_hidden_layers)

    loaded_data = load_data(dataset_name)

    classifier = MethodGraphBertNodeClassification(bert_config, loaded_data)
    # ---- set to false to run faster ----
    classifier.spy_tag = True
    classifier.max_epoch = max_epoch
    classifier.lr = lr
    return classifier





class MethodGraphBertNodeClassification(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    spy_tag = True

    def __init__(self, config, data):
        super(MethodGraphBertNodeClassification, self).__init__(config)
        self.config = config
        self.bert = MethodGraphBert(config)
        self.res_h = torch.nn.Linear(config.x_size, config.hidden_size)
        self.res_y = torch.nn.Linear(config.x_size, config.y_size)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.y_size)
        self.init_weights()
        self.data = data

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, idx=None):
        residual_h, residual_y = self.residual_term()
        if idx is not None:
            if residual_h is None:
                outputs = self.bert(raw_features[idx], wl_role_ids[idx], init_pos_ids[idx], hop_dis_ids[idx],
                                    residual_h=None)
            else:
                outputs = self.bert(raw_features[idx], wl_role_ids[idx], init_pos_ids[idx], hop_dis_ids[idx],
                                    residual_h=residual_h[idx])
                residual_y = residual_y[idx]
        else:
            if residual_h is None:
                outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, residual_h=None)
            else:
                outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, residual_h=residual_h)

        sequence_output = 0
        for i in range(self.config.k + 1):
            sequence_output += outputs[0][:, i, :]
        sequence_output /= float(self.config.k + 1)

        labels = self.cls_y(sequence_output)

        if residual_y is not None:
            labels += residual_y

        return F.log_softmax(labels, dim=1)

    # 残差项
    def residual_term(self):
        if self.config.residual_type == 'none':
            return None, None
        elif self.config.residual_type == 'raw':
            return self.res_h(self.data['X']), self.res_y(self.data['X'])
        elif self.config.residual_type == 'graph_raw':
            return torch.spmm(self.data['A'], self.res_h(self.data['X'])), torch.spmm(self.data['A'],
                                                                                      self.res_y(self.data['X']))



# 3. 客户端本地训练
def client_update(client_model):
    optimizer = optim.Adam(client_model.parameters(), lr=client_model.lr, weight_decay=client_model.weight_decay)
    accuracy = EvaluateAcc('', '')

    _train = client_model.data['idx_train']
    _val = client_model.data['idx_val']
    _test = client_model.data['idx_test']

    for epoch in range(max_epoch):

        client_model.train()
        optimizer.zero_grad()

        output = client_model.forward(client_model.data['raw_embeddings'], client_model.data['wl_embedding'], client_model.data['int_embeddings'],
                              client_model.data['hop_embeddings'], client_model.data['idx_train'])

        loss_train = F.cross_entropy(output, client_model.data['y'][client_model.data['idx_train']])
        accuracy.data = {'true_y': client_model.data['y'][client_model.data['idx_train']], 'pred_y': output.max(1)[1]}

        loss_train.backward()
        optimizer.step()

        client_model.eval()
        output = client_model.forward(client_model.data['raw_embeddings'], client_model.data['wl_embedding'], client_model.data['int_embeddings'],
                              client_model.data['hop_embeddings'], client_model.data['idx_val'])

        loss_val = F.cross_entropy(output, client_model.data['y'][client_model.data['idx_val']])
        accuracy.data = {'true_y': client_model.data['y'][client_model.data['idx_val']], 'pred_y': output.max(1)[1]}

    client_model.eval()
    # todo: 2024/10/23/21/06/00理解是什么意思
    output = client_model.forward(client_model.data['raw_embeddings'], client_model.data['wl_embedding'], client_model.data['int_embeddings'],
                          client_model.data['hop_embeddings'], client_model.data['idx_test'])
    loss = F.cross_entropy(output, client_model.data['y'][client_model.data['idx_test']])
    accuracy.data = {'true_y': client_model.data['y'][client_model.data['idx_test']], 'pred_y': output.max(1)[1]}
    acc_test = accuracy.evaluate()
    acc = acc_test.item()
    print("本次通信的客户端精度 = ", acc)



    return client_model.state_dict()  # 返回模型参数状态


# 4. 联邦平均算法聚合模型参数
def federated_averaging(global_model, client_states):
    global_state = global_model.state_dict()
    for key in global_state.keys():
        global_state[key] = torch.stack([client_state[key].float() for client_state in client_states]).mean(dim=0)
    global_model.load_state_dict(global_state)


# 5. 联邦学习主过程
def federated_training():
    global_model = Classifier("globalgraph")
    client_models = []
    accuracy = EvaluateAcc('', '')

    for i in range(config.client_num):
        client_models.append(Classifier("subgraph" + str(i + 1)))
    print("客户端构造完成")

    # 通信轮次
    for round in range(config.round):
        print("开始第 ", round, " 次通信")
        client_states = []
        for i in range(config.client_num):
            print("客户端", i+1)
            # 每个客户端加载全局模型的参数
            client_models[i].load_state_dict(global_model.state_dict())
            # 在客户端本地数据上训练模型
            client_state = client_update(client_models[i])
            client_states.append(client_state)

        # 聚合客户端模型参数
        federated_averaging(global_model, client_states)
        print(f"Round {round + 1} completed.")

    # 全局模型评估
    global_model.load_state_dict(client_models[0].state_dict())
    output = global_model.forward(global_model.data['raw_embeddings'], global_model.data['wl_embedding'], global_model.data['int_embeddings'],
                          global_model.data['hop_embeddings'], global_model.data['idx_test'])

    accuracy.data = {'true_y': global_model.data['y'][global_model.data['idx_test']], 'pred_y': output.max(1)[1]}
    acc_test = accuracy.evaluate()
    acc = acc_test.item()
    print("average test accuracy: {:.4f}".format(acc))






# 执行联邦学习
federated_training()
