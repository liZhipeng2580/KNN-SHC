import math

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class KNN_SHC:
    def __init__(self, X_L, Y_L, X_U, Y_True, lambda_param, k_neighbors=5, max_iterations=10):
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
        self.clusters = {i: X_L[Y_L == i] for i in range(self.k)}
        self.labels = {i: Y_L[Y_L == i] for i in range(self.k)}
        self.centroids = self.compute_centroids(self.clusters)
        self.knn = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        self.knn.fit(X_L, Y_L)
        self.iterations = 0

    def compute_centroids(self, clusters_data):
        """
        计算每个簇的质心，仅使用有标签的样本。
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
        # print(len( self.centroids.items()))
        D = sum([np.linalg.norm(x - centroid) for centroid in self.centroids.values()])
        for i, centroid in self.centroids.items():
            distance = np.linalg.norm(x - centroid)
            distance_confidence = (D - distance) / D
            knn_neighbors = self.knn.kneighbors([x], return_distance=False)[0]
            knn_neighbors_labels = np.concatenate((self.Y_L, self.Y_P))[knn_neighbors]
            N = np.sum(knn_neighbors_labels == i)
            knn_confidence = N / len(knn_neighbors)
            probabilities[i] = self.lambda_param * distance_confidence + (1 - self.lambda_param) * knn_confidence
        return probabilities

    def reassign_clusters(self, X, Y):
        new_clusters = {}
        new_labels = {}
        print(f"迭代 {self.iterations}: 样本数量={len(X)}, 标签数量={len(Y)}")

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

        print(f"迭代 {self.iterations}: 新簇数量={len(new_clusters)}, 新标签数量={len(new_labels)}")

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
            min_distance = centroid_distances[centroid_prediction]

            knn_prediction = self.knn.predict([x])[0]
            knn_neighbors = self.knn.kneighbors([x], return_distance=False)[0]
            knn_neighbors_labels = np.concatenate((self.Y_L, self.Y_P))[knn_neighbors]

            same_cluster_neighbors = np.sum(knn_neighbors_labels == centroid_prediction)
            if centroid_prediction == knn_prediction and same_cluster_neighbors > 3:
                self.X_P = np.vstack([self.X_P, x])
                self.Y_U[i] = centroid_prediction
                pseudo_labels.append(centroid_prediction)
            else:
                self.Y_U[i] = None

        self.Y_P = np.array(pseudo_labels)
        print(f"一共{len(self.X_U)}个无标签样本，生成了 {len(pseudo_labels)} 个伪标签。")

    def update_centroids(self):
        """
        更新质心矩阵，结合有标签和伪标签样本。
        """
        if self.X_P.size > 0:
            self.centroids = self.compute_centroids(self.clusters)
            print(f"更新质心之后{len(self.centroids.items())}")

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

    def iterative_clustering(self):
        """
        迭代更新簇分配，计算经验损失和聚类损失，直到损失函数不再减小或没有新的簇产生。
        """
        prev_loss = float('inf')
        err_list = []
        sse_list = []

        self.iterations = 0
        # 重新分配簇

        # 构建样本集
        combined_X = np.vstack((self.X_L, self.X_U))

        # 构建标签集
        combined_Y = np.concatenate((self.Y_L, self.Y_U))
        self.clusters, self.labels = self.reassign_clusters(combined_X, combined_Y)

        print(f"迭代 {self.iterations}: 样本数量={len(combined_X)}")
        print(f"迭代 {self.iterations}: 标签数量={len(combined_Y)}")
        # self.clusters, self.labels = self.reassign_clusters(combined_X, np.concatenate((self.Y_L, self.Y_P)))

        while self.iterations < self.max_iterations:
            print(f"迭代 {self.iterations}: 样本个数={len(self.clusters.items())}")

            # # 计算新的经验损失和SSE
            # Y_pred = self.knn.predict(self.X_L)
            # err = self.calculate_err(self.Y_L, Y_pred)
            # sse = self.calculate_sse(self.clusters)
            # err_list.append(err)
            # sse_list.append(sse)
            #
            # # 归一化经验损失和SSE
            # if np.sum(err_list) != 0:
            #     err_normalized = err / np.sum(err_list)
            # else:
            #     err_normalized = 0
            # if np.sum(sse_list) != 0:
            #     sse_normalized = sse / np.sum(sse_list)
            # else:
            #     sse_normalized = 0
            #
            # # 计算新的损失函数
            # new_loss = self.lambda_param * err_normalized + (1 - self.lambda_param) * sse_normalized
            #
            # print(f"迭代 {self.iterations}: new_loss={new_loss}, prev_loss={prev_loss}, 簇数量={len(self.clusters)}")
            #
            # if new_loss >= prev_loss:
            #     print("因 new_loss >= prev_loss 而退出")
            #     break
            #
            # prev_loss = new_loss
            self.iterations += 1
            has_changed = False

            # 初始化新簇的字典和新簇的标签
            new_clusters = {}
            new_labels = {}
            new_cluster_ids = []
            new_cluster_id = max(self.clusters.keys()) + 1  # 新簇的初始ID

            print(f"簇分裂前:簇个数: {len(self.clusters)},簇类别: {self.labels.items()}")

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
                print(f"标签 {combined_labels}")
                print(f"标签 {unique_labels} :{counts}")
                # 如果簇内有多个不同的标签
                if len(filtered_labels) > 1:
                    majority_label = filtered_labels[np.argmax(filtered_counts)]  # 主要标签（出现次数最多的标签）
                    print(f"簇 {cluster_id} 含有多个类的样本: {filtered_labels}，主要标签是:{majority_label}")

                    # 遍历每个标签
                    for label in filtered_labels:
                        if math.isnan(label):
                            continue
                        if label != majority_label:
                            print(f"不同标签是:{label}")
                            # 获取当前标签的索引
                            label_indices = np.where(combined_labels == label)[0]

                            # 从原簇中找到这些标签对应的数据点
                            new_cluster_points = cluster_points[label_indices]

                            # 如果新簇中有数据点
                            if len(new_cluster_points) > 0:
                                print(
                                    f"将类 {label} 样本从簇 {cluster_id} 分裂到新簇 {new_cluster_id} 样本个数{len(new_cluster_points)}")
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
                # else:
                #     for x in cluster_points:
                #         knn_neighbors = self.knn.kneighbors([x], return_distance=False)[0]
                #         knn_neighbors_labels = np.concatenate((self.Y_L, self.Y_P))[knn_neighbors]
                #         if np.sum(knn_neighbors_labels == cluster_id) <= 3:
                #             # 根据近邻情况将x转移到其它簇
                #             cluster_probabilities = self.calculate_cluster_probabilities(x)
                #             new_cluster_id = max(cluster_probabilities, key=cluster_probabilities.get)
                #             if new_cluster_id != cluster_id:
                #                 print(f"将样本从簇 {cluster_id} 转移到簇 {new_cluster_id}")
                #                 self.clusters[new_cluster_id] = np.vstack([self.clusters[new_cluster_id], x])
                #                 self.labels[new_cluster_id] = np.append(self.labels[new_cluster_id], cluster_id)
                #                 self.clusters[cluster_id] = self.clusters[cluster_id][
                #                     ~np.all(self.clusters[cluster_id] == x, axis=1)]
                #                 self.labels[cluster_id] = self.labels[cluster_id][self.labels[cluster_id] != cluster_id]
                #                 has_changed = True

            # 更新原簇集合
            self.clusters.update(new_clusters)
            self.labels.update(new_labels)
            # 更新质心，根据伪标签和有标签样本更新质心
            self.update_centroids()

            print(f"重新分簇前:簇个数: {len(self.clusters)},簇类别: {self.labels.items()}")

            # 重新分簇
            # 构建样本集
            combined_X = np.vstack((self.X_L, self.X_U))
            # 构建标签集
            combined_Y = np.concatenate((self.Y_L, self.Y_U))
            self.clusters, self.labels = self.reassign_clusters(combined_X, combined_Y)
            self.update_centroids()
            print(f"重新分簇后:簇个数: {len(self.clusters)},簇类别: {self.labels.items()}")

            # 删除空簇
            empty_clusters = [cluster_id for cluster_id, points in self.clusters.items() if len(points) == 0]
            for cluster_id in empty_clusters:
                print(f"删除空簇 {cluster_id}")
                del self.clusters[cluster_id]
                del self.labels[cluster_id]
                if cluster_id in self.centroids:
                    del self.centroids[cluster_id]

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
                    if len(true_labels) > 0 & len(pseudo_labels) > 0:
                        acc = np.mean(np.concatenate((true_labels, pseudo_labels)) == cluster_id)
                        if acc < avg_acc:
                            print(f"新簇 {cluster_id} 的准确率低于平均值 {avg_acc}，删除")

                            del self.clusters[cluster_id]
                            del self.labels[cluster_id]

            if not has_changed:
                print("因没有变化而退出")
                break

    def final_clustering(self):
        """
        根据最终的质心对所有样本进行分簇。
        """
        combined_X = np.vstack((self.X_L, self.X_U))
        self.clusters, self.labels = self.reassign_clusters(combined_X, np.concatenate((self.Y_L, self.Y_P)))

    def run(self):
        """
        执行KNN-SHC算法，包括预测伪标签、更新质心、迭代更新和最终簇划分。
        """
        # 步骤1: 初始化簇和质心矩阵
        # 已在 __init__ 中完成

        # 步骤2: 预测伪标签
        self.predict_pseudo_labels()

        # 步骤3: 更新质心矩阵
        self.update_centroids()

        # 步骤4: 迭代更新簇
        self.iterative_clustering()

        # 步骤5: 最终簇划分
        self.final_clustering()

        return self.clusters


# 加载Iris数据集
data = load_breast_cancer()
X, Y = data.data, data.target

# 归一化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分有标签和无标签数据
X_L, X_U, Y_L, Y_U = train_test_split(X, Y, test_size=0.8, random_state=42)

lambda_param = 0.9

# 初始化KNN-SHC算法并运行
knn_shc = KNN_SHC(X_L, Y_L, X_U, Y_U, lambda_param)
clusters = knn_shc.run()

# 打印最终簇划分结果
for cluster_id, cluster_points in clusters.items():
    print(f"簇 {cluster_id}: {len(cluster_points)} 个点")

# 计算NMI和RI
all_labels = np.concatenate((Y_L, [-1] * len(X_U)))
pred_labels = np.zeros_like(all_labels)

for cluster_id, cluster_points in clusters.items():
    for point in cluster_points:
        idx = np.where(np.all(X == point, axis=1))[0]
        pred_labels[idx] = cluster_id

# NMI和RI指标计算
nmi = normalized_mutual_info_score(Y, pred_labels)
ri = adjusted_rand_score(Y, pred_labels)

print(f"迭代次数: {knn_shc.iterations}")
print(f"NMI: {nmi}")
print(f"RI: {ri}")
