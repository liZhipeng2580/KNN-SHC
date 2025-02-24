# -*- coding: utf-8 -*-

import logging
import os
from datetime import datetime

from sklearn.datasets import load_breast_cancer, load_wine, load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import pandas as pd
from main import KNN_SHC
# from torch_geometric.datasets import WebKB

# 创建数据集加载字典
datasets = {
    # 'breast_cancer': load_breast_cancer,
    #  'wine': load_wine,
    # 'iris': load_iris,
     'ecoli': lambda: fetch_openml(name='ecoli', version=1),
    # 'glass': lambda: fetch_openml(name='glass', version=1),
    # 'pendigits': lambda: fetch_openml(name='pendigits', version=1),
    #  'satimage': lambda: fetch_openml(name='satimage', version=1),
    # 'optdigits': lambda: fetch_openml(name='optdigits', version=1),
    # # 'webkb': lambda: fetch_openml("webkb", "cornell"),
    # 'penbased': lambda: fetch_openml(name='penbased', version='active'),
    # 'segment': lambda: fetch_openml(name='segment', version=1)
}


def setup_logger(dataset_name):
    # 获取当前日期，格式为 YYYY-MM-DD
    current_date = datetime.now().strftime('%Y-%m-%d-%H')
    log_dir = os.path.join('logs', current_date)  # 创建以日期为名的子文件夹

    # 确保日志文件夹存在
    os.makedirs(log_dir, exist_ok=True)
    log_filename =  os.path.join(log_dir, f"{dataset_name}_{datetime.now().strftime('%H-%M-%S')}.log")
    logger = logging.getLogger(dataset_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def compute_pred_labels_and_metrics(self, X, Y,Y_L, logger):
    clusters = self.clusters
    all_labels = np.concatenate((Y_L, [-1] * (X.shape[0] - len(Y_L))))
    pred_labels = np.zeros_like(all_labels)

    for cluster_id, cluster_points in clusters.items():
        combined_labels = self.labels[cluster_id]
        unique_labels, counts = np.unique(combined_labels, return_counts=True)
        valid_indices = ~np.isnan(unique_labels)
        filtered_labels = unique_labels[valid_indices]
        if len(filtered_labels) == 0:
            logger.info(f"当前簇内没有有效的标签: {cluster_id}")
            continue
        filtered_counts = counts[valid_indices]
        majority_label = filtered_labels[np.argmax(filtered_counts)]
        for point in cluster_points:
            idx = np.where(np.all(X == point, axis=1))[0]
            pred_labels[idx] = majority_label

    # 计算NMI和RI
    nmi = normalized_mutual_info_score(Y, pred_labels)
    ri = adjusted_rand_score(Y, pred_labels)
    logger.info(f"NMI: {nmi}")
    logger.info(f"RI: {ri}")

    return pred_labels, nmi, ri


def run_knn_shc_on_datasets(datasets):
    for dataset_name, load_function in datasets.items():
        logger = setup_logger(dataset_name)
        logger.info(f"开始处理数据集: {dataset_name}")
        # dataset = WebKB(root, name)
        # 加载数据集

        if dataset_name == 'webkb':
            categories = ['comp.graphics', 'rec.autos', 'sci.med', 'soc.religion.christian']  # 示例类别
            data = fetch_20newsgroups(categories=categories, remove=('headers', 'footers', 'quotes'))
            X, Y = data.data, data.target
        elif dataset_name == 'penbased':
            data = np.array(pd.read_csv("./dataset/{}.csv".format(dataset_name), header=None))
            X = data[:, 1:]
            Y = np.asarray(data[:, 0], dtype=int)
            data = fetch_20newsgroups()
        else:
            data = load_function()
            X, Y = data.data, data.target

        # 归一化数据
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)

        # 划分有标签和无标签数据
        X_L, X_U, Y_L, Y_U = train_test_split(X, Y, test_size=0.8)
        Y_L = np.array(Y_L).astype(int)
        Y_U = np.array(Y_U).astype(int)
        lambda_param = 0.5

        # 初始化KNN-SHC算法并运行
        knn_shc = KNN_SHC(X_L, Y_L, X_U, Y_U, lambda_param, dataset_name)
        self = knn_shc.run()

        logger.info("X_L shape: %s", X_L.shape)
        logger.info("Y_L shape: %s", Y_L.shape)
        logger.info("Unique labels in Y_L: %s", np.unique(Y_L))
        logger.info("Label counts: %s", np.bincount(Y_L))

        clusters = self.clusters
        for cluster_id, cluster_points in clusters.items():
            logger.info(f"簇 {cluster_id}: {len(cluster_points)} 个点")

        # 计算预测标签和指标
        compute_pred_labels_and_metrics(self, X, Y, Y_L, logger)


# 执行
run_knn_shc_on_datasets(datasets)
