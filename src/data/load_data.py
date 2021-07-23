import numpy as np
import torch
from loguru import logger
from scipy.sparse import coo_matrix


def encode_onehot(labels):
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_cora_data(data_path):

    logger.info("Loading Cora dataset")

    idx_features_labels = np.genfromtxt(f"{str(data_path)}/cora.content", dtype=np.dtype(str))
    features = np.array(idx_features_labels[:, 1: -1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{str(data_path)}/cora.cites", dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                     shape=(labels.shape[0], labels.shape[0]), dtype=np.float32).toarray()

    # Symmetric adjacency matrix
    adj = adj + np.multiply(adj.T, adj.T > adj) - np.multiply(adj, adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + np.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = torch.FloatTensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(adj):
    row_sum = np.array(adj.sum(1))
    r_inv_sqrt = np.power(row_sum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    return r_mat_inv_sqrt.dot(adj).dot(r_mat_inv_sqrt)


def normalize_features(features):

    row_sum = np.array(features.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    return r_mat_inv.dot(features)
