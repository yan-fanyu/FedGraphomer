'''
Concrete IO class for a specific dataset
'''
import time

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv
import pickle
from sklearn.model_selection import train_test_split

# 加载数据集
class DatasetLoader(dataset):
    c = 0.15
    k = 5
    data = None
    batch_size = None

    dataset_source_folder_path = None
    dataset_name = None

    load_all_tag = False
    compute_s = False

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(DatasetLoader, self).__init__(dName, dDescription)

    def load_hop_wl_batch(self):
        # print('Load WL Dictionary')
        f = open('./result/WL/' + self.dataset_name, 'rb')
        wl_dict = pickle.load(f)
        f.close()

        # print('Load Hop Distance Dictionary')
        f = open('./result/Hop/hop_' + self.dataset_name + '_' + str(self.k), 'rb')
        hop_dict = pickle.load(f)
        f.close()

        # print('Load Subgraph Batches')
        f = open('./result/Batch/' + self.dataset_name + '_' + str(self.k), 'rb')
        batch_dict = pickle.load(f)
        f.close()

        return hop_dict, wl_dict, batch_dict

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def load(self):
        """Load citation network dataset (cora only for now)"""
        # print('Loading {} dataset...'.format(self.dataset_name))

        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

        one_hot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

        idx_map = {j: i for i, j in enumerate(idx)}
        index_id_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(one_hot_labels.shape[0], one_hot_labels.shape[0]),
                            dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        eigen_adj = None
        if self.compute_s:
            eigen_adj = self.c * inv((sp.eye(adj.shape[0]) - (1 - self.c) * self.adj_normalize(adj)).toarray())

        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))


        arr = idx_features_labels[:, [0, -1]]

        # print(arr)
        # time.sleep(20)
        labels = arr[:, 1]
        # 首次划分：80% 用于训练集+验证集，20% 用于测试集
        train_val_indices, test_indices = train_test_split(np.arange(len(arr)), test_size=0.2, stratify=labels, random_state=42)
        # 第二次划分：训练集+验证集中的数据再划分成 60% 训练集，20% 验证集
        train_indices, val_indices = train_test_split(train_val_indices, test_size=0.25, stratify=labels[train_val_indices], random_state=42)

        # 输出三个数据集的下标数组
        idx_train = train_indices
        idx_val = val_indices
        idx_test = test_indices

        # np.savetxt(self.dataset_name + "_train.txt", idx_train, fmt='%d')
        # np.savetxt(self.dataset_name + "_val.txt", idx_val, fmt='%d')
        # np.savetxt(self.dataset_name + "_test.txt", idx_test, fmt='%d')
        # print("**************************** 保存OK ************************")

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(one_hot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        if self.load_all_tag:
            hop_dict, wl_dict, batch_dict = self.load_hop_wl_batch()
            raw_feature_list = []
            role_ids_list = []
            position_ids_list = []
            hop_ids_list = []
            for node in idx:

                node_index = idx_map[node]
                # print(node, "\t", node_index)
                neighbors_list = batch_dict[node]
                # print(neighbors_list)
                # time.sleep(2)
                # raw_feature 1433*1
                raw_feature = [features[node_index].tolist()]

                # print(node, "\t", len(neighbors_list))

                role_ids = [wl_dict[node]]
                position_ids = range(len(neighbors_list) + 1)
                hop_ids = [0]
                # 遍历邻居节点和其亲密度 [(686532, 0.0647), (31349, 0.06479), (10531, 0.06442), (1129442, 0.06012), (31353, 0.0599), (43698, 0.0460), (194617, 0.01759)]
                for neighbor, intimacy_score in neighbors_list:

                    neighbor_index = idx_map[neighbor]
                    raw_feature.append(features[neighbor_index].tolist())
                    role_ids.append(wl_dict[neighbor])
                    if neighbor in hop_dict[node]:
                        hop_ids.append(hop_dict[node][neighbor])
                    else:
                        hop_ids.append(99)
                raw_feature_list.append(raw_feature)
                # print(len(raw_feature))

                role_ids_list.append(role_ids)
                position_ids_list.append(position_ids)
                hop_ids_list.append(hop_ids)

            # todo: raw_embeddings = 2708*8*1433, 第一个存放本节点的特征向量，其余存放7个上下文节点的特征
            raw_embeddings = torch.FloatTensor(raw_feature_list)
            # todo: wl_embedding 2708 * 8  怎么来的
            wl_embedding = torch.LongTensor(role_ids_list)
            # todo: hop_embeddings 2708 * 8 怎么来的
            hop_embeddings = torch.LongTensor(hop_ids_list)
            # todo: int_embeddings 2708 * 8 怎么来的
            int_embeddings = torch.LongTensor(position_ids_list)


        else:
            raw_embeddings, wl_embedding, hop_embeddings, int_embeddings = None, None, None, None

        res = {'X': features, 'A': adj, 'S': eigen_adj, 'index_id_map': index_id_map, 'edges': edges_unordered,
               'raw_embeddings': raw_embeddings, 'wl_embedding': wl_embedding, 'hop_embeddings': hop_embeddings,
               'int_embeddings': int_embeddings, 'y': labels, 'idx': idx, 'idx_train': idx_train, 'idx_test': idx_test,
               'idx_val': idx_val}

        return res
