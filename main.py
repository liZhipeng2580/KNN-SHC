import math
import logging
import os
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset


class KNN_SHC:
    def __init__(self, X_L, Y_L, X_U, Y_True, lambda_param,dataset_name, k_neighbors=3, max_iterations=1, ):
        self.X_L = X_L
        self.Y_L = Y_L
        self.X_U = X_U
        self.Y_U = np.empty(len(self.X_U))
        self.Y_True = Y_True  # 无标签样本的真实标签，用于计算准确率
        self.lambda_param = lambda_param
        self.k_neighbors = k_neighbors
        self.max_iterations = max_iterations
        self.X_P = np.empty((0, X_L.shape[1]))  # 伪标签样本集
        self.Y_P = np.array([])
        self.k = len(np.unique(Y_L))  # 簇的数量
        self.clusters = {label: X_L[Y_L == label] for label in np.unique(Y_L)}
        self.labels = {label: Y_L[Y_L == label] for label in np.unique(Y_L)}
        self.centroids = self.compute_centroids(self.clusters)
        self.knn = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        self.knn.fit(X_L, Y_L)
        self.iterations = 0
        self.Y_L_Pre = np.empty(len(self.Y_L))
        self.dataset_name = dataset_name
        self.logger = self.setup_logger()

    def compute_centroids(self, clusters_data):
        """
        计算每个簇的质心，使用有标签的样本和伪标签的样本
        """
        centroids = {}
        for cluster_id, samples in clusters_data.items():
            # 获取该簇的标签
            cluster_labels = self.labels[cluster_id]

            # 筛选出有标签的样本
            labeled_samples = [samples[i] for i in range(len(samples)) if not np.isnan(cluster_labels[i])]

            if labeled_samples:
                # 计算有标签样本的质心
                centroid = np.mean(labeled_samples, axis=0)
                centroids[cluster_id] = centroid

        return centroids

    def calculate_cluster_probabilities(self, x):
        probabilities = {}

        D = sum([np.linalg.norm(x - centroid) for centroid in self.centroids.values()])
        for i, centroid in self.centroids.items():
            distance = np.linalg.norm(x - centroid)
            distance_confidence = (D - distance) / D
            probabilities[i] = self.lambda_param * distance_confidence
        return probabilities

    def calculate_cluster_probabilities_end(self, x):
        probabilities = {}

        D = sum([np.linalg.norm(x - centroid) for centroid in self.centroids.values()])
        for i, centroid in self.centroids.items():
            distance = np.linalg.norm(x - centroid)
            distance_confidence = (D - distance) / D
            knn_neighbors = self.knn.kneighbors([x], return_distance=False)[0]
            knn_neighbors_samples = np.vstack((self.X_L, self.X_P))[knn_neighbors]
            # 计算N，表示K个近邻中属于簇p_i的样本数量
            N = sum(any(np.array_equal(sample, cluster_point) for cluster_point in self.clusters[i]) for sample in
                    knn_neighbors_samples)
            knn_confidence = N / len(knn_neighbors)
            probabilities[i] = self.lambda_param * distance_confidence + (1 - self.lambda_param) * knn_confidence
        return probabilities

    def reassign_clusters(self, X, Y):
        new_clusters = {}
        new_labels = {}
        self.logger.info(f"迭代 {self.iterations}: 样本数量={len(X)}, 标签数量={len(Y)}")
        self.logger.info(f"迭代 {self.iterations}: 分簇时簇质心集合={self.centroids.values()}")
        for x, y in zip(X, Y):
            cluster_probabilities = self.calculate_cluster_probabilities(x)
            assigned_cluster = max(cluster_probabilities, key=cluster_probabilities.get)
            if assigned_cluster not in new_clusters:
                new_clusters[assigned_cluster] = []
                new_labels[assigned_cluster] = []

            new_clusters[assigned_cluster].append(x)

            # 如果样本有标签或伪标签，则将其标签加入新标签列表
            if y is not None:  # 假设None表示无标签样本
                new_labels[assigned_cluster].append(y)

        self.logger.info(f"迭代 {self.iterations}: 新簇数量={len(new_clusters)}, 新标签数量={len(new_labels)}")

        # 返回非空的簇和标签
        return {i: np.array(points) for i, points in new_clusters.items() if len(points) > 0}, \
            {i: np.array(labels) for i, labels in new_labels.items() if len(labels) > 0}

    def reassign_clusters_end(self, X, Y):
        new_clusters = {}
        new_labels = {}
        self.logger.info(f"迭代 {self.iterations}: 样本数量={len(X)}, 标签数量={len(Y)}")

        for x, y in zip(X, Y):
            cluster_probabilities = self.calculate_cluster_probabilities_end(x)
            assigned_cluster = max(cluster_probabilities, key=cluster_probabilities.get)

            if assigned_cluster not in new_clusters:
                new_clusters[assigned_cluster] = []
                new_labels[assigned_cluster] = []

            new_clusters[assigned_cluster].append(x)

            # 如果样本有标签或伪标签，则将其标签加入新标签列表
            if y is not None:  # 假设None表示无标签样本
                new_labels[assigned_cluster].append(y)

        self.logger.info(f"迭代 {self.iterations}: 新簇数量={len(new_clusters)}, 新标签数量={len(new_labels)}")

        # 返回非空的簇和标签
        return {i: np.array(points) for i, points in new_clusters.items() if len(points) > 0}, \
            {i: np.array(labels) for i, labels in new_labels.items() if len(labels) > 0}

    def predict_pseudo_labels(self):
        """
        预测无标签样本的伪标签，基于质心和KNN的一致性，并且增加距离阈值和邻居比例限制。
        """
        pseudo_labels = []
        for i, x in enumerate(self.X_U):

            centroid_distances = {i: np.linalg.norm(x - self.centroids[i]) for i in self.centroids}
            centroid_prediction = min(centroid_distances, key=centroid_distances.get)
            predict_pseudo_knn = KNeighborsClassifier(n_neighbors=5)
            predict_pseudo_knn.fit(self.X_L, self.Y_L)
            knn_prediction = predict_pseudo_knn.predict([x])[0]
            knn_neighbors = predict_pseudo_knn.kneighbors([x], return_distance=False)[0]
            knn_neighbors_labels = np.concatenate((self.Y_L, self.Y_P))[knn_neighbors]

            same_cluster_neighbors = np.sum(knn_neighbors_labels == centroid_prediction)

            # 设置一致性阈值
            if centroid_prediction == knn_prediction and same_cluster_neighbors > 3:
                self.X_P = np.vstack([self.X_P, x])
                self.Y_U[i] = centroid_prediction
                pseudo_labels.append(centroid_prediction)
            else:
                self.Y_U[i] = None

        self.Y_P = np.array(pseudo_labels)
        self.logger.info(f"一共{len(self.X_U)}个无标签样本，生成了 {len(pseudo_labels)} 个伪标签。")
        self.logger.info(f"生成伪标签后，无样本标签为：{self.Y_U}")
    def retrain_knn_with_pseudo_labels(self):
        """
        重新训练 KNN 分类器，包括有标签样本和伪标签样本。
        """
        # 合并有标签样本和伪标签样本
        X_combined = np.vstack((self.X_L, self.X_P))
        Y_combined = np.concatenate((self.Y_L, self.Y_P))

        # 创建并训练 KNN 分类器
        self.knn.fit(X_combined, Y_combined)

    def update_centroids(self):
        """
        更新质心矩阵，结合有标签和伪标签样本。
        """
        if self.X_P.size > 0:
            self.centroids = self.compute_centroids(self.clusters)
            self.logger.info(f"更新质心之后{len(self.centroids.items())}")

    def compute_Y_L_pred(self):
        """
        根据簇内的多数标签生成有标签样本的预测标签。
        """
        pred_labels = np.copy(self.Y_L)  # 初始化预测标签为 -1

        # 创建一个映射，用于快速查找 X 中的样本在 X_L 中的索引
        X_to_XL_indices = {tuple(x): i for i, x in enumerate(self.X_L)}

        for cluster_id, cluster_points in self.clusters.items():
            # 获取簇内所有标签
            combined_labels = self.labels[cluster_id]
            # 统计标签的唯一性及其计数
            unique_labels, counts = np.unique(combined_labels, return_counts=True)
            # 过滤掉 NaN 标签
            valid_indices = ~np.isnan(unique_labels)
            filtered_labels = unique_labels[valid_indices]
            filtered_counts = counts[valid_indices]

            # 找到多数标签
            if filtered_counts.size > 0:
                majority_label = filtered_labels[np.argmax(filtered_counts)]

                # 将多数标签分配给簇中的每个点
                for point in cluster_points:
                    # 检查点是否在 X_L 中
                    point_tuple = tuple(point)
                    if point_tuple in X_to_XL_indices:
                        idx = X_to_XL_indices[point_tuple]
                        pred_labels[idx] = majority_label

        return pred_labels

    def calculate_err(self, Y_L, Y_pred):
        """
        计算经验损失，即预测标签与真实标签之间的误差。
        """
        return np.mean(Y_L != Y_pred)

    def calculate_sse(self, clusters):
        """
        计算无监督聚类损失，即样本与簇质心之间的距离总和。
        """
        sse = 0
        for cluster_id, cluster_points in clusters.items():
            sse += np.sum([np.linalg.norm(x - self.centroids[cluster_id]) ** 2 for x in cluster_points])
        return sse

    def total_loss(self, Err_t, SSE_t):
        """计算总损失函数 (L)"""
        Err_value = self.calculate_err(self.Y_L, self.Y_L_Pre)
        SSE_value = self.calculate_sse(self.clusters)

        # 归一化损失
        Err_t.append(Err_value)
        SSE_t.append(SSE_value)

        sum_Err_t = np.sum(Err_t)
        sum_SSE_t = np.sum(SSE_t)

        # 避免除以零
        Err_prime_j = Err_value / sum_Err_t if sum_Err_t != 0 else 0
        SSE_prime_j = SSE_value / sum_SSE_t if sum_SSE_t != 0 else 0

        # 计算总损失
        total_loss_value = self.lambda_param * Err_prime_j + (1 - self.lambda_param) * SSE_prime_j
        return total_loss_value

    def iterative_clustering(self):
        """
        迭代更新簇分配，计算经验损失和聚类损失，直到损失函数不再减小或没有新的簇产生。
        """
        # 存储损失函数
        Err_t = []
        SSE_t = []
        last_loss = 0.00
        current_loss = 0.00
        self.iterations = 0
        # 重新分配簇

        # 构建样本集
        combined_X = np.vstack((self.X_L, self.X_U))

        # 构建标签集
        combined_Y = np.concatenate((self.Y_L, self.Y_U))
        self.clusters, self.labels = self.reassign_clusters_end(combined_X, combined_Y)
        acc = self.calculate_label_accuracy()
        self.logger.info(f"分配簇的ACC为: {acc}")

        # 重新计算质心
        self.update_centroids()

        self.logger.info(f"迭代 {self.iterations}: 样本数量={len(combined_X)}")
        self.logger.info(f"迭代 {self.iterations}: 标签数量={len(combined_Y)}")
        # self.clusters, self.labels = self.reassign_clusters(combined_X, np.concatenate((self.Y_L, self.Y_P)))

        while 1 == 1:
            self.logger.info(f"迭代 {self.iterations}: 样本个数={len(self.clusters.items())}")

            self.iterations += 1
            has_changed = False

            # 初始化新簇的字典和新簇的标签
            new_clusters = {}
            new_labels = {}
            new_cluster_ids = []
            new_cluster_id = max(self.clusters.keys()) + 1  # 新簇的初始ID

            self.logger.info(f"簇分裂前:簇个数: {len(self.clusters)},簇类别: {self.labels.items()}")

            # 遍历现有簇
            for cluster_id, cluster_points in list(self.clusters.items()):
                # 簇内标签样本
                combined_labels = self.labels[cluster_id]
                # 找到簇中所有唯一标签及其计数
                unique_labels, counts = np.unique(combined_labels, return_counts=True)
                # 排除 NaN 标签
                valid_indices = ~np.isnan(unique_labels)
                filtered_labels = unique_labels[valid_indices]
                filtered_counts = counts[valid_indices]
                self.logger.info(f"{cluster_id}簇内全部标签 {combined_labels}")
                self.logger.info(f"{cluster_id}簇内不同标签 {filtered_labels} :{filtered_counts}")
                # 如果簇内有多个不同的标签
                if len(filtered_labels) > 0:
                    majority_label = filtered_labels[np.argmax(filtered_counts)]  # 主要标签（出现次数最多的标签）
                    self.logger.info(f"簇 {cluster_id} 含有多个类的样本: {filtered_labels}，主要标签是:{majority_label}")

                    # 遍历每个标签
                    for label in filtered_labels:
                        if math.isnan(label):
                            continue
                        if label != majority_label:
                            self.logger.info(f"不同标签是:{label}")
                            # 获取当前标签的索引
                            label_indices = np.where(combined_labels == label)[0]

                            # 从原簇中找到这些标签对应的数据点
                            new_cluster_points = cluster_points[label_indices]

                            # 如果新簇中有数据点
                            if len(new_cluster_points) > 0:
                                self.logger.info(
                                    f"将类 {label} 样本从簇 {cluster_id} 分裂到新簇 {new_cluster_id} 样本个数{len(new_cluster_points)}")
                                self.logger.info(f"样本为:{new_cluster_points}")
                                new_clusters[new_cluster_id] = new_cluster_points
                                new_labels[new_cluster_id] = combined_labels[label_indices]
                                new_cluster_ids.append(new_cluster_id)

                                # 更新原始簇，移除已被分离的点
                                remaining_indices = ~np.isin(np.arange(len(cluster_points)), label_indices)
                                self.clusters[cluster_id] = cluster_points[remaining_indices]
                                self.labels[cluster_id] = combined_labels[remaining_indices]

                                # 更新新簇ID
                                new_cluster_id += 1
                                has_changed = True
                else:
                    for x in cluster_points:
                        knn_neighbors = self.knn.kneighbors([x], return_distance=False)[0]
                        knn_neighbors_labels = np.concatenate((self.Y_L, self.Y_P))[knn_neighbors]
                        if np.sum(knn_neighbors_labels == cluster_id) <= 3:
                            # 根据近邻情况将x转移到其它簇
                            self.logger.info(f"将要将样本从簇 {cluster_id} 转移到簇 {new_cluster_id}")
                            cluster_probabilities = self.calculate_cluster_probabilities(x)
                            new_cluster_id = max(cluster_probabilities, key=cluster_probabilities.get)
                            if new_cluster_id != cluster_id:
                                self.logger.info(f"将样本从簇 {cluster_id} 转移到簇 {new_cluster_id}")
                                self.clusters[new_cluster_id] = np.vstack([self.clusters[new_cluster_id], x])
                                self.labels[new_cluster_id] = np.append(self.labels[new_cluster_id], cluster_id)
                                self.clusters[cluster_id] = self.clusters[cluster_id][
                                    ~np.all(self.clusters[cluster_id] == x, axis=1)]
                                self.labels[cluster_id] = self.labels[cluster_id][self.labels[cluster_id] != cluster_id]
                                has_changed = True

            # 更新原簇集合
            self.clusters.update(new_clusters)
            self.labels.update(new_labels)
            # 更新质心，根据伪标签和有标签样本更新质心
            self.update_centroids()

            self.logger.info(f"重新分簇前:簇个数: {len(self.clusters)},簇类别: {self.labels.items()}")

            # 重新分簇
            # 构建样本集
            combined_X = np.vstack((self.X_L, self.X_U))
            # 构建标签集
            combined_Y = np.concatenate((self.Y_L, self.Y_U))

            self.clusters, self.labels = self.reassign_clusters_end(combined_X, combined_Y)
            self.logger.info(f"重新分簇后:簇个数: {len(self.clusters)},簇类别: {self.labels.items()}")
            acc = self.calculate_label_accuracy()
            self.update_centroids()
            self.logger.info(f"分配簇的ACC为: {acc}")


            # 计算簇的准确率
            avg_acc = np.mean([
                np.mean(np.concatenate(
                    (self.Y_L[np.all(np.isin(self.X_L, points), axis=1)],
                     self.Y_P[np.all(np.isin(self.X_P, points), axis=1)])) == cluster_id)
                for cluster_id, points in self.clusters.items()
                if len(points) > 0
            ])

            # 仅删除新簇，并将样本重新分配
            for cluster_id in new_cluster_ids:
                if cluster_id in self.clusters:
                    true_labels = self.Y_L[np.all(np.isin(self.X_L, self.clusters[cluster_id]), axis=1)]
                    pseudo_labels = self.Y_P[np.all(np.isin(self.X_P, self.clusters[cluster_id]), axis=1)]
                    if len(true_labels) > 0 and len(pseudo_labels) > 0:
                        acc = np.mean(np.concatenate((true_labels, pseudo_labels)) == cluster_id)
                        if acc < avg_acc:
                            self.logger.info(f"新簇 {cluster_id} 的准确率低于平均值 {avg_acc}，删除")
                            del self.clusters[cluster_id]
                            del self.labels[cluster_id]
                            del self.centroids[cluster_id]
            if not has_changed:
                self.logger.info("因没有变化而退出")
                break
            # 计算损失函数
            self.Y_L_Pre = self.compute_Y_L_pred()
            num_errors = np.sum(self.Y_L != self.Y_L_Pre)
            self.logger.info("不相等标签个数是：" + str(num_errors))
            current_loss = self.total_loss(Err_t, SSE_t)
            self.logger.info("损失函数：" + str(current_loss))
            if last_loss != 0.00 and current_loss > last_loss:
                self.logger.info("损失函数增大，退出循环")
                break
            else:
                last_loss = current_loss

    def final_clustering(self):
        """
        根据最终的质心对所有样本进行分簇。
        """
        combined_X = np.vstack((self.X_L, self.X_U))
        self.clusters, self.labels = self.reassign_clusters_end(combined_X, np.concatenate((self.Y_L, self.Y_U)))
        acc = self.calculate_label_accuracy()
        self.logger.info(f"最终分簇后:簇个数: {len(self.clusters)},簇类别: {self.labels.items()}")
        self.logger.info(f"分配簇的ACC为: {acc}")

    def calculate_pseudo_label_accuracy(self):
        """
        计算伪标签的生成准确性，排除 NaN 标签。
        """
        # 确保 Y_U 和 Y_True 是 numpy 数组
        self.Y_U = np.array(self.Y_U)
        self.Y_True = np.array(self.Y_True)

        # 排除 NaN 标签
        valid_indices = ~np.isnan(self.Y_U)

        # 提取有效的伪标签和真实标签
        pseudo_labels = self.Y_U[valid_indices]
        true_labels = self.Y_True[valid_indices]
        # 打印有效标签的数量
        self.logger.info(f"无标签数量: {len(self.Y_U)}")
        # 打印有效标签的数量
        self.logger.info(f"有效伪标签数量: {len(pseudo_labels)}")
        self.logger.info(f"伪标签比例: {len(pseudo_labels) / len(self.Y_U)}")
        # 计算准确性
        accuracy = np.mean(pseudo_labels == true_labels)
        self.logger.info(f"伪标签生成的准确性: {accuracy:.4f}")
        return accuracy

    # 计算分配簇的准确率
    def calculate_label_accuracy(self):
        """
        计算分配簇后样本的准确性
        """
        # 确保 Y_U 和 Y_True 是 numpy 数组
        self.Y_U = np.array(self.Y_U)
        self.Y_True = np.array(self.Y_True)

        # 创建副本用于预测标签
        predicted_labels = np.full(self.Y_U.shape, -1)  # 用 -1 初始化

        # 遍历现有簇
        for cluster_id, cluster_points in list(self.clusters.items()):
            # 获取当前簇中所有样本的伪标签
            cluster_labels = self.labels[cluster_id]

            # 排除NaN值
            valid_labels = cluster_labels[~np.isnan(cluster_labels)]

            # 如果簇中没有有效的标签，则跳过
            if len(valid_labels) == 0:
                self.logger.info(f"当前簇内没有有效的标签: {cluster_id}")
                continue

            # 找到当前簇的主要标签（出现频率最高的标签）
            unique, counts = np.unique(valid_labels, return_counts=True)
            most_common_label = unique[np.argmax(counts)]

            # 更新 NaN 标签
            for point in cluster_points:
                idx = np.where(np.all(self.X_U == point, axis=1))[0]
                predicted_labels[idx] = most_common_label

        # 计算标签准确性
        correct_predictions = np.sum(predicted_labels == self.Y_True)
        total_samples = len(self.Y_True)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        return accuracy

    def run(self):
        """
        执行KNN-SHC算法，包括预测伪标签、更新质心、迭代更新和最终簇划分。
        """
        # 步骤1: 初始化簇和质心矩阵
        # 已在 __init__ 中完成

        # 步骤2: 预测伪标签
        self.predict_pseudo_labels()

        # 步骤3: 重新训练KNN
        self.retrain_knn_with_pseudo_labels()

        # 步骤3: 更新质心矩阵
        self.update_centroids()

        # 步骤4: 迭代更新簇
        self.iterative_clustering()

        # 步骤5: 最终簇划分
        self.final_clustering()

        # 步骤6：计算伪标签的准确率
        self.calculate_pseudo_label_accuracy()

        return self

    def setup_logger(self):
        current_date = datetime.now().strftime('%Y-%m-%d')
        log_dir = os.path.join('logs', current_date)  # 创建以日期为名的子文件夹

        # 确保日志文件夹存在
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"{self.dataset_name}_{datetime.now().strftime('%H-%M-%S')}.log")

        # 创建自定义logger
        logger = logging.getLogger(self.dataset_name)
        logger.setLevel(logging.INFO)  # 设置日志级别

        # 避免重复添加handlers
        if not logger.handlers:
            # 创建文件处理器和控制台处理器
            file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
            console_handler = logging.StreamHandler()

            # 设置日志格式
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # 添加处理器到logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger


# # 加载Iris数据集
# data = load_breast_cancer()
# # data = load_wine()
# # data = load_iris()
#
# # data = fetch_openml(name='ecoli', version=1)
# # data = fetch_openml(name='glass', version=1)
# # data = fetch_openml(name='pendigits', version=1)
# #  data = fetch_openml(name='satimage', version=1)
# # data = fetch_openml(name='optdigits', version=1)
# # data = load_dataset("webkb", "cornell")
# # data = fetch_openml(name='penbased', version='active')
# # data = fetch_openml(name='segment', version=1)
# X, Y = data.data, data.target
#
# # 归一化数据
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# label_encoder = LabelEncoder()
# Y = label_encoder.fit_transform(Y)
# # 划分有标签和无标签数据
# X_L, X_U, Y_L, Y_U = train_test_split(X, Y, test_size=0.7)
# Y_L = np.array(Y_L)
# Y_L = Y_L.astype(int)
# Y_U = np.array(Y_U)
# Y_U = Y_U.astype(int)
# lambda_param = 0.5
#
# # 初始化KNN-SHC算法并运行
# knn_shc = KNN_SHC(X_L, Y_L, X_U, Y_U, lambda_param,data.)
# self = knn_shc.run()
# self.logger.info("X_L shape: %s", X_L.shape)
# self.logger.info("Y_L shape: %s", Y_L.shape)
# self.logger.info("Unique labels in Y_L: %s", np.unique(Y_L))
# self.logger.info("Label counts: %s", np.bincount(Y_L.astype(int)))
# clusters = self.clusters
# # 打印最终簇划分结果
# for cluster_id, cluster_points in clusters.items():
#     self.logger.info(f"簇 {cluster_id}: {len(cluster_points)} 个点")
#
# # 计算NMI和RI
# all_labels = np.concatenate((Y_L, [-1] * len(X_U)))
# pred_labels = np.zeros_like(all_labels)
#
# for cluster_id, cluster_points in clusters.items():
#     # 簇内标签样本
#     combined_labels = self.labels[cluster_id]
#     # 找到簇中所有唯一标签及其计数
#     unique_labels, counts = np.unique(combined_labels, return_counts=True)
#     # 排除 NaN 标签
#     valid_indices = ~np.isnan(unique_labels)
#     filtered_labels = unique_labels[valid_indices]
#     if len(filtered_labels) == 0:
#         self.logger.info(f"当前簇内没有有效的标签: {cluster_id}")
#         continue
#     filtered_counts = counts[valid_indices]
#     majority_label = filtered_labels[np.argmax(filtered_counts)]
#     for point in cluster_points:
#         idx = np.where(np.all(X == point, axis=1))[0]
#         pred_labels[idx] = majority_label
#
# # NMI和RI指标计算
# nmi = normalized_mutual_info_score(Y, pred_labels)
# ri = adjusted_rand_score(Y, pred_labels)
#
# self.logger.info(f"迭代次数: {knn_shc.iterations}")
# self.logger.info(f"NMI: {nmi}")
# self.logger.info(f"RI: {ri}")

