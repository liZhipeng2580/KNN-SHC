# -*- coding: utf-8 -*-
import time
from datetime import datetime

import numpy as np
from collections import defaultdict

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, fetch_openml


from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score, rand_score
from sklearn.preprocessing import StandardScaler


class Cluster:
    def __init__(self, label, is_new=False):
        self.samples = []  # 存储样本 (特征, 标签)
        self.centroid = None  # 簇质心
        self.label = label  # 簇标签
        self.is_new = is_new  # 标记是否为分裂产生的新簇

    def update_centroid(self):
        """更新簇质心"""
        if not self.samples:
            self.centroid = None
            return
        X = np.array([x for x, _ in self.samples])
        self.centroid = X.mean(axis=0)

    def update_label(self):
        """根据有标签样本更新簇标签"""
        labeled_samples = [y for _, y in self.samples if y is not None]
        if not labeled_samples:
            return
        counts = defaultdict(int)
        for y in labeled_samples:
            counts[y] += 1
        self.label = max(counts, key=counts.get)

    def get_labeled_samples(self):
        """获取有标签样本"""
        return [(x, y) for x, y in self.samples if y is not None]


class KNN_SHC:
    def __init__(self, lambda_param=0.5, k_neighbors=5, max_iters=100):
        self.lambda_param = lambda_param
        self.k_neighbors = k_neighbors
        self.max_iters = max_iters
        self.clusters = []  # 簇集合
        self.D_P = {'X': [], 'Y': []}  # 伪标签数据集
        self.X_combined = None  # 合并后的特征数据（X_L + D_P）
        self.Y_combined = None  # 合并后的标签数据（Y_L + Y_P）
        self.loss_history = []  # 损失记录

    def initialize_clusters(self, X_L, Y_L):
        """步骤1：初始化簇结构"""
        self.clusters = []
        for label in np.unique(Y_L):
            cluster = Cluster(label)
            indices = np.where(Y_L == label)[0]
            for i in indices:
                cluster.samples.append((X_L[i], Y_L[i]))
            cluster.update_centroid()
            self.clusters.append(cluster)

    def _update_combined_data(self, X_L, Y_L):
        """更新合并数据集"""
        X_pseudo = np.array(self.D_P['X']) if self.D_P['X'] else np.empty((0, X_L.shape[1]))
        Y_pseudo = np.array(self.D_P['Y']) if self.D_P['Y'] else np.empty(0)
        self.X_combined = np.vstack([X_L, X_pseudo])
        self.Y_combined = np.concatenate([Y_L, Y_pseudo])

    def generate_pseudo_labels(self, X_U, X_L, Y_L):
        """步骤2：生成伪标签"""
        self._update_combined_data(X_L, Y_L)
        new_X, new_Y = [], []

        # 构建KNN模型（仅使用有标签数据）
        if len(X_L) == 0:
            return
        knn = NearestNeighbors(n_neighbors=self.k_neighbors)
        knn.fit(X_L)

        for x in X_U:
            # 质心预测
            centroid_probs = []
            valid_clusters = [c for c in self.clusters if c.centroid is not None]

            for cluster in valid_clusters:
                # 计算距离可信度
                total_distance = sum(np.linalg.norm(x - c.centroid) for c in valid_clusters)
                dist = np.linalg.norm(x - cluster.centroid)
                dist_trust = (total_distance - dist) / total_distance if total_distance != 0 else 0

                # 计算KNN可信度
                _, indices = knn.kneighbors([x])
                knn_labels = Y_L[indices[0]]
                N = np.sum(knn_labels == cluster.label)
                knn_trust = N / self.k_neighbors

                # 综合概率
                prob = self.lambda_param * dist_trust + (1 - self.lambda_param) * knn_trust
                centroid_probs.append((prob, cluster.label))

            if centroid_probs:
                max_prob = max(centroid_probs, key=lambda x: x[0])
                pred_label = max_prob[1]

                # 检查近邻一致性
                _, indices = knn.kneighbors([x])
                if np.sum(Y_L[indices[0]] == pred_label) > 3:
                    new_X.append(x)
                    new_Y.append(pred_label)

        # 更新伪标签集
        if new_X:
            self.D_P['X'].extend(np.array(new_X))
            self.D_P['Y'].extend(np.array(new_Y))
            current_ratio = len(new_X) / len(X_U) if len(X_U) != 0 else 0
            print(f"本次迭代生成伪标签: {len(new_X)} 个（占比: {current_ratio:.2%}）")
        self._update_combined_data(X_L, Y_L)

    def _calculate_probability(self, x, cluster):
        """计算样本x属于簇的概率（式3.6）"""
        # 距离可信度
        valid_clusters = [c for c in self.clusters if c.centroid is not None]
        if not valid_clusters or cluster.centroid is None:
            return 0.0

        total_distance = sum(np.linalg.norm(x - c.centroid) for c in valid_clusters)
        dist = np.linalg.norm(x - cluster.centroid)
        dist_trust = (total_distance - dist) / total_distance if total_distance != 0 else 0

        # 近邻可信度
        if self.X_combined.size == 0:
            return dist_trust

        knn = NearestNeighbors(n_neighbors=self.k_neighbors)
        knn.fit(self.X_combined)
        _, indices = knn.kneighbors([x])
        neighbor_labels = self.Y_combined[indices[0]]
        N = np.sum(neighbor_labels == cluster.label)
        knn_trust = N / self.k_neighbors

        return self.lambda_param * dist_trust + (1 - self.lambda_param) * knn_trust

    def update_clusters(self, X_L, Y_L, X_U):
        """步骤3：更新簇分配"""
        # 清空现有簇样本
        for cluster in self.clusters:
            cluster.samples = []

        # 分配所有样本
        X_all = np.vstack([X_L, self.D_P['X'], X_U])
        Y_all = np.concatenate([Y_L, self.D_P['Y'], np.full(len(X_U), None)])

        for x, y in zip(X_all, Y_all):
            probs = []
            for cluster in self.clusters:
                if cluster.centroid is None:
                    continue
                prob = self._calculate_probability(x, cluster)
                probs.append((prob, cluster))

            if probs:
                selected_cluster = max(probs, key=lambda x: x[0])[1]
                selected_cluster.samples.append((x, y))

        # 更新质心和标签
        for cluster in self.clusters:
            cluster.update_centroid()
            cluster.update_label()

    def split_clusters(self):
        """修正后的簇分裂逻辑"""
        new_clusters = []
        for cluster in self.clusters.copy():
            labeled_samples = cluster.get_labeled_samples()
            if not labeled_samples:
                continue
            labels = [y for _, y in labeled_samples]
            unique_labels = np.unique(labels)

            if len(unique_labels) > 1:
                for label in unique_labels:
                    if label == cluster.label:
                        continue
                    new_cluster = Cluster(label, is_new=True)  # 标记为新簇
                    new_samples = [(x, y) for x, y in labeled_samples if y == label]
                    new_cluster.samples = new_samples
                    new_cluster.update_centroid()
                    new_clusters.append(new_cluster)
                cluster.samples = [s for s in cluster.samples if s[1] == cluster.label]
                cluster.update_centroid()
        self.clusters.extend(new_clusters)
        return len(new_clusters) > 0

    def transfer_samples(self):
        """修正后的样本转移逻辑（解决数组比较问题）"""
        transferred = False

        # 使用合并后的数据（X_combined）构建KNN
        if len(self.X_combined) < self.k_neighbors:
            return False
        knn = NearestNeighbors(n_neighbors=self.k_neighbors)
        knn.fit(self.X_combined)

        for cluster in self.clusters.copy():
            if cluster.label is None:
                continue

            # 遍历副本，避免修改原列表导致问题
            for sample in list(cluster.samples):  # 使用list()创建副本
                x, y = sample
                # 获取近邻标签
                _, indices = knn.kneighbors([x])
                neighbor_labels = self.Y_combined[indices[0]]

                # 统计异类数量
                foreign_count = sum(1 for label in neighbor_labels if label != cluster.label)
                if foreign_count > 3:
                    # 寻找最佳目标簇
                    probs = []
                    for c in self.clusters:
                        if c.centroid is None:
                            continue
                        prob = self._calculate_probability(x, c)
                        probs.append((prob, c))

                    if probs:
                        best_cluster = max(probs, key=lambda x: x[0])[1]
                        if best_cluster != cluster:
                            # 精确匹配并删除样本
                            for idx, s in enumerate(cluster.samples):
                                s_x, s_y = s
                                if np.array_equal(s_x, x) and s_y == y:
                                    del cluster.samples[idx]
                                    best_cluster.samples.append((x, y))
                                    transferred = True
                                    break
        return transferred

    def prune_clusters(self):
        """根据新簇准确率是否低于全局平均进行剪枝"""
        if len(self.clusters) <= 2:
            return  # 至少保留两个簇

        # 计算所有簇的准确率
        all_accuracies = []
        new_cluster_info = []  # 存储新簇的索引及其准确率

        for idx, cluster in enumerate(self.clusters):
            # 获取有标签和伪标签的样本
            labeled = cluster.get_labeled_samples()
            if not labeled:
                acc = 0.0
            else:
                correct = sum(1 for (_, y) in labeled if y == cluster.label)
                acc = correct / len(labeled)
            all_accuracies.append(acc)

            # 记录新簇信息
            if cluster.is_new:
                new_cluster_info.append((idx, acc))

        # 计算全局平均准确率
        global_avg = sum(all_accuracies) / len(all_accuracies)

        # 确定待删除的新簇索引（准确率 < 全局平均）
        to_remove = [idx for idx, acc in new_cluster_info if acc < global_avg]

        # 逆序删除以避免索引错位
        for idx in reversed(sorted(to_remove)):
            if len(self.clusters) > 2:
                del self.clusters[idx]

    def compute_loss(self, X_L, Y_L):
        """计算归一化损失（式3.9）"""
        # 经验误差
        Y_pred = []
        for x in X_L:
            probs = [self._calculate_probability(x, c) for c in self.clusters]
            if not probs:
                Y_pred.append(-1)
                continue
            Y_pred.append(self.clusters[np.argmax(probs)].label)
        error = np.mean([1 if y_true != y_pred else 0 for y_true, y_pred in zip(Y_L, Y_pred)])

        # SSE计算
        sse = 0.0
        for cluster in self.clusters:
            if cluster.centroid is None:
                continue
            for x, _ in cluster.samples:
                sse += np.linalg.norm(x - cluster.centroid) ** 2

        # 归一化
        self.loss_history.append((error, sse))
        if len(self.loss_history) < 2:
            return error + sse

        total_error = sum(e for e, _ in self.loss_history)
        total_sse = sum(s for _, s in self.loss_history)
        norm_error = error / total_error if total_error != 0 else 0
        norm_sse = sse / total_sse if total_sse != 0 else 0
        return norm_error + norm_sse

    def visualize_clusters(self):
        """可视化当前簇状态"""
        plt.figure(figsize=(12, 8))
        color_map = {}
        current_color = 0

        # 绘制所有簇的样本和质心
        for cluster in self.clusters:
            samples = cluster.samples
            if not samples:
                continue

            # 确定颜色（优先使用簇标签，无标签则用新颜色）
            labels = [y for _, y in samples]
            unique_labels = np.unique(labels)
            if len(unique_labels) == 0:
                # 无标签样本统一用灰色
                color = (0.5, 0.5, 0.5)
            else:
                # 使用簇标签对应的颜色
                if cluster.label not in color_map:
                    color_map[cluster.label] = current_color
                    current_color += 1
                color = color_map[cluster.label]

            # 绘制样本点
            xs = [x[0] for x, _ in samples]
            ys = [x[1] for x, _ in samples]
            plt.scatter(xs, ys, c=color, alpha=0.6,
                        label=f'Cluster {cluster.label}' if cluster.label is not None else 'Unlabeled')

        # 绘制质心
        for cluster in self.clusters:
            if cluster.centroid is not None:
                cx, cy = cluster.centroid
                plt.scatter(cx, cy, marker='x', s=150, color='red', label='Centroid')

        # 设置图例和标题
        plt.title("Current Clustering State")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

    def fit(self, X_L, Y_L, X_U):
        """主训练流程"""
        # 步骤1-2：初始化+生成伪标签
        self.initialize_clusters(X_L, Y_L)
        self.generate_pseudo_labels(X_U, X_L, Y_L)
        self.update_clusters(X_L, Y_L, X_U)

        prev_loss = float('inf')
        cluster_changed = True
        iter_count = 0

        while cluster_changed and iter_count < self.max_iters:
            print(f"次数: {iter_count}")
            # self.visualize_clusters()
            # 步骤4：簇操作
            split_flag = self.split_clusters()
            transfer_flag = self.transfer_samples()
            self.update_clusters(X_L, Y_L, X_U)
            self.prune_clusters()
            # self.visualize_clusters()
            # 计算当前状态
            current_loss = self.compute_loss(X_L, Y_L)
            cluster_changed = split_flag or transfer_flag
            print(f"当前损失函数: {current_loss}")
            # 终止条件
            if current_loss > prev_loss:
                break
            prev_loss = current_loss
            iter_count += 1

    def evaluate(self, X_test, y_true):
        """评估方法"""
        y_pred = []
        for x in X_test:
            probs = [self._calculate_probability(x, c) for c in self.clusters]
            if not probs:
                y_pred.append(-1)
                continue
            y_pred.append(self.clusters[np.argmax(probs)].label)
        return {
            "NMI": normalized_mutual_info_score(y_true, y_pred),
            "RI": rand_score(y_true, y_pred)
        }


# 示例使用
if __name__ == "__main__":
    # 加载数据集
    cancer = fetch_openml(name='ecoli', version=1)
    X, y = cancer.data, cancer.target
    y = cancer.target.to_numpy()

    # 数据预处理
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 获取真实类别数
    true_clusters = len(np.unique(y))
    print(f"数据集信息：样本数={len(X)}, 特征数={X.shape[1]}, 真实类别数={true_clusters}")

    # 初始化结果存储
    results = {
        'SHC': {'NMI': [], 'RI': [], 'Time': []},
        'KMeans': {'NMI': [], 'RI': [], 'Time': []}
    }
    results_df = pd.DataFrame(columns=['Run', 'SHC_NMI', 'SHC_RI', 'SHC_Time',
                                       'KMeans_NMI', 'KMeans_RI', 'KMeans_Time'])

    # 进行实验
    num_runs = 50
    for run in range(num_runs):
        print(f"\n=== 第 {run + 1}/{num_runs} 次实验 ===")

        # 数据分割
        try:
            X_L, X_U, y_L, y_U = train_test_split(
                X, y,
                test_size=0.9,
                stratify=y,
                random_state=run
            )

            # if len(np.unique(y_L)) < true_clusters:
            #     print(f"警告：标签不完整，跳过本次运行")
            #     continue
        except Exception as e:
            print(f"数据分割错误: {str(e)}")
            continue

        # 运行KNN-SHC
        shc_metrics = {'NMI': np.nan, 'RI': np.nan}
        try:
            shc_start = time.time()
            model = KNN_SHC(lambda_param=0.5, k_neighbors=5, max_iters=20)
            model.fit(X_L, y_L, X_U)
            shc_time = time.time() - shc_start
            shc_metrics = model.evaluate(X_U, y_U)
            print(f"当前运行指标 -> NMI: {shc_metrics['NMI']:.4f}, RI: {shc_metrics['RI']:.4f}")
        except Exception as e:
            print(f"KNN-SHC 错误: {str(e)}")

        # 运行KMeans
        kmeans_metrics = {'NMI': np.nan, 'RI': np.nan}
        try:
            kmeans_start = time.time()
            kmeans = KMeans(n_clusters=true_clusters, random_state=run)
            kmeans_pred = kmeans.fit_predict(X_U)
            kmeans_time = time.time() - kmeans_start
            kmeans_metrics = {
                'NMI': normalized_mutual_info_score(y_U, kmeans_pred),
                'RI': rand_score(y_U, kmeans_pred)
            }
        except Exception as e:
            print(f"KMeans 错误: {str(e)}")

        # 记录结果
        new_row = {
            'Run': run + 1,
            'SHC_NMI': shc_metrics['NMI'],
            'SHC_RI': shc_metrics['RI'],
            'SHC_Time': shc_time if 'shc_time' in locals() else np.nan,
            'KMeans_NMI': kmeans_metrics['NMI'],
            'KMeans_RI': kmeans_metrics['RI'],
            'KMeans_Time': kmeans_time if 'kmeans_time' in locals() else np.nan
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    # 保存结果到Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"clustering_results_{timestamp}.xlsx"

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # 详细结果
        results_formatted = results_df.round(4)
        results_formatted.to_excel(writer, sheet_name='详细结果', index=False)

        # 统计摘要
        stats = pd.DataFrame({
            'Metric': ['NMI', 'RI', 'Time(s)'],
            'SHC_Mean': [
                results_df['SHC_NMI'].mean(),
                results_df['SHC_RI'].mean(),
                results_df['SHC_Time'].mean()
            ],
            'SHC_Std': [
                results_df['SHC_NMI'].std(),
                results_df['SHC_RI'].std(),
                results_df['SHC_Time'].std()
            ],
            'KMeans_Mean': [
                results_df['KMeans_NMI'].mean(),
                results_df['KMeans_RI'].mean(),
                results_df['KMeans_Time'].mean()
            ],
            'KMeans_Std': [
                results_df['KMeans_NMI'].std(),
                results_df['KMeans_RI'].std(),
                results_df['KMeans_Time'].std()
            ]
        })
        stats.to_excel(writer, sheet_name='统计摘要', index=False)

        # 格式设置
        workbook = writer.book
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })

        for sheet in writer.sheets:
            worksheet = writer.sheets[sheet]
            for col_num, value in enumerate(results_formatted.columns):
                worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(0, 0, 8)
            worksheet.set_column(1, len(results_formatted.columns) - 1, 12)

    print(f"\n实验结果已保存至: {filename}")

    # 可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.boxplot([results_df['SHC_NMI'], results_df['KMeans_NMI']],
                labels=['SHC', 'KMeans'])
    plt.title('Normalized Mutual Information')

    plt.subplot(1, 2, 2)
    plt.boxplot([results_df['SHC_RI'], results_df['KMeans_RI']],
                labels=['SHC', 'KMeans'])

