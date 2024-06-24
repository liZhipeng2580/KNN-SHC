import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import accuracy_score


def calculate_centroids(X, labels, num_clusters):
    """
    计算每个簇的质心

    参数：
    X: 样本数据，形状为 (n_samples, n_features)
    labels: 每个样本的标签，形状为 (n_samples,)
    num_clusters: 簇的数量

    返回值：
    centroids: 质心矩阵，形状为 (num_clusters, n_features)
    """
    centroids = []
    for i in range(num_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        else:
            centroids.append(np.zeros(X.shape[1]))
    return np.array(centroids)


def knn_predict(X, X_labeled, y_labeled, X_pseudo, y_pseudo, k=5):
    """
    使用KNN预测无标签数据的簇

    参数：
    X: 无标签数据，形状为 (n_samples, n_features)
    X_labeled: 有标签数据，形状为 (n_labeled_samples, n_features)
    y_labeled: 有标签数据的标签，形状为 (n_labeled_samples,)
    X_pseudo: 伪标签数据，形状为 (n_pseudo_samples, n_features)
    y_pseudo: 伪标签数据的标签，形状为 (n_pseudo_samples,)
    k: KNN的最近邻数量

    返回值：
    predictions: 无标签数据的预测簇标签，形状为 (n_samples,)
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(np.vstack([X_labeled, X_pseudo]), np.hstack([y_labeled, y_pseudo]))
    return knn.predict(X)


def assign_clusters(X, centroids, lambda_param):
    """
    根据质心和距离分配簇标签

    参数：
    X: 样本数据，形状为 (n_samples, n_features)
    centroids: 质心矩阵，形状为 (num_clusters, n_features)
    lambda_param: 权重参数

    返回值：
    labels: 分配的簇标签，形状为 (n_samples,)
    """
    distances = pairwise_distances(X, centroids)
    total_distance = np.sum(distances, axis=1)
    probabilities = []

    for i in range(distances.shape[1]):
        distance_prob = lambda_param * (total_distance - distances[:, i]) / total_distance
        neighbor_prob = (1 - lambda_param) * np.mean(distances[:, i])
        probabilities.append(distance_prob + neighbor_prob)

    probabilities = np.array(probabilities).T
    return np.argmax(probabilities, axis=1)


def compute_pseudo_labels(X_u, X_labeled, y_labeled, centroids, k=5):
    """
    计算无标签数据的伪标签

    参数：
    X_u: 无标签数据，形状为 (n_samples, n_features)
    X_labeled: 有标签数据，形状为 (n_labeled_samples, n_features)
    y_labeled: 有标签数据的标签，形状为 (n_labeled_samples,)
    centroids: 质心矩阵，形状为 (num_clusters, n_features)
    k: KNN的最近邻数量

    返回值：
    pseudo_labels: 伪标签数据和对应的标签列表
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_labeled, y_labeled)

    pseudo_labels = []
    for i, x in enumerate(X_u):
        # 根据质心分配的簇标签
        centroid_distances = np.linalg.norm(x - centroids, axis=1)
        centroid_label = np.argmin(centroid_distances)

        # 根据KNN预测的簇标签
        knn_distances, knn_indices = knn.kneighbors([x], n_neighbors=k)
        knn_labels = y_labeled[knn_indices[0]]
        knn_label = np.argmax(np.bincount(knn_labels))

        # 检查是否一致，且近邻中大于3个属于该簇
        if centroid_label == knn_label and np.sum(knn_labels == centroid_label) > 3:
            pseudo_labels.append((x, centroid_label))

    return pseudo_labels


def split_clusters(X, labels, num_clusters):
    """
    对包含多个类的簇进行分裂

    参数：
    X: 样本数据，形状为 (n_samples, n_features)
    labels: 每个样本的簇标签，形状为 (n_samples,)
    num_clusters: 簇的数量

    返回值：
    new_labels: 分裂后的簇标签，形状为 (n_samples,)
    """
    new_labels = labels.copy()
    for cluster in range(num_clusters):
        cluster_points = X[labels == cluster]
        cluster_labels = labels[labels == cluster]

        unique_labels = np.unique(cluster_labels)
        if len(unique_labels) > 1:
            for unique_label in unique_labels:
                new_cluster_label = max(new_labels) + 1
                new_labels[(labels == cluster) & (cluster_labels == unique_label)] = new_cluster_label

    return new_labels


def transfer_samples(X, labels, centroids, k=5):
    """
    根据K近邻情况将样本转移到其他簇

    参数：
    X: 样本数据，形状为 (n_samples, n_features)
    labels: 每个样本的簇标签，形状为 (n_samples,)
    centroids: 质心矩阵，形状为 (num_clusters, n_features)
    k: KNN的最近邻数量

    返回值：
    new_labels: 转移后的簇标签，形状为 (n_samples,)
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, labels)
    neighbors = knn.kneighbors(X, return_distance=False)

    new_labels = labels.copy()
    for i, x in enumerate(X):
        majority_label = np.argmax(np.bincount(labels[neighbors[i]]))
        if majority_label != labels[i]:
            new_labels[i] = majority_label

    return new_labels


def compute_loss(X_L, y_L, X_U, labels, centroids, lambda_param):
    """
    计算损失函数

    参数：
    X_L: 有标签数据，形状为 (n_labeled_samples, n_features)
    y_L: 有标签数据的标签，形状为 (n_labeled_samples,)
    X_U: 无标签数据，形状为 (n_unlabeled_samples, n_features)
    labels: 所有样本的簇标签，形状为 (n_samples,)
    centroids: 质心矩阵，形状为 (num_clusters, n_features)
    lambda_param: 权重参数

    返回值：
    loss: 损失值
    """
    y_L_pred = labels[:len(X_L)]
    err = np.mean(y_L != y_L_pred)
    sse = np.sum([np.linalg.norm(x - centroids[label]) for x, label in zip(X_L, labels)])
    return lambda_param * err + (1 - lambda_param) * sse


def compute_accuracy(y_true, y_pred):
    """
    计算每个簇的预测准确率

    参数：
    y_true: 真实标签
    y_pred: 预测标签

    返回值：
    accuracy: 准确率
    """
    return accuracy_score(y_true, y_pred)


def knn_shc(D_L, D_U, lambda_param=0.5, k=5, max_iter=100):
    """
    KNN-SHC算法

    参数：
    D_L: 有标签数据集 (X_L, y_L)
    D_U: 无标签数据集 (X_U, None)
    lambda_param: 权重参数
    k: KNN的最近邻数量
    max_iter: 最大迭代次数

    返回值：
    labels: 最终的簇标签
    """
    X_L, y_L = D_L
    X_U, _ = D_U
    X_P, y_P = np.array([]), np.array([])

    # 步骤1：用有标签数据初始化簇
    num_clusters = len(np.unique(y_L))
    centroids = calculate_centroids(X_L, y_L, num_clusters)

    prev_loss = float('inf')

    for iteration in range(max_iter):
        # 步骤2：使用质心和KNN预测无标签数据的簇
        pseudo_labels = compute_pseudo_labels(X_U, X_L, y_L, centroids, k=k)

        if not pseudo_labels:
            break

        X_P = np.array([x for x, _ in pseudo_labels])
        y_P = np.array([label for _, label in pseudo_labels])

        # 步骤3：用有标签和伪标签数据更新质心
        centroids = calculate_centroids(np.vstack([X_L, X_P]), np.hstack([y_L, y_P]), num_clusters)

        # 步骤4：重新分配簇标签和更新标签
        all_data = np.vstack([X_L, X_U])
        labels = assign_clusters(all_data, centroids, lambda_param)

        # 分裂包含多个类的簇
        labels = split_clusters(all_data, labels, num_clusters)

        # 根据KNN多数投票转移样本
        labels = transfer_samples(all_data, labels, centroids, k=k)

        # 计算损失
        loss = compute_loss(X_L, y_L, X_U, labels, centroids, lambda_param)

        print(f"Iteration {iteration}, Loss: {loss:.4f}")

        if loss >= prev_loss:
            break

        prev_loss = loss

        # 计算每个簇的预测准确率
        accuracies = []
        for cluster in range(num_clusters):
            cluster_labels = labels[:len(X_L)][labels[:len(X_L)] == cluster]
            cluster_true_labels = y_L[labels[:len(X_L)] == cluster]
            if len(cluster_true_labels) > 0:
                acc = compute_accuracy(cluster_true_labels, cluster_labels)
                accuracies.append(acc)

        mean_accuracy = np.mean(accuracies)

        # 删除准确率低于平均值的新簇
        for cluster in range(num_clusters, max(labels) + 1):
            cluster_labels = labels[:len(X_L)][labels[:len(X_L)] == cluster]
            cluster_true_labels = y_L[labels[:len(X_L)] == cluster]
            if len(cluster_true_labels) > 0:
                acc = compute_accuracy(cluster_true_labels, cluster_labels)
                if acc < mean_accuracy:
                    labels[labels == cluster] = -1  # 将该簇标记为无效簇

    return labels


# # 使用示例
D_L = (np.array([
    [1, 2], [2, 3], [3, 4], [6, 7],
    [8, 8], [9, 9], [10, 10], [11, 11],
    [5, 5], [4, 4], [7, 7], [6, 6],
    [3, 3], [2, 2], [1, 1], [0, 0]
]), np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]))

D_U = (np.array([
    [1.5, 2.5], [3.5, 4.5], [6.5, 7.5], [8.5, 9.5],
    [10.5, 11.5], [4.5, 5.5], [5.5, 6.5], [2.5, 3.5],
    [0.5, 1.5], [7.5, 8.5], [9.5, 10.5], [6.5, 5.5],
    [3.5, 2.5], [2.5, 1.5], [0.5, 0.5], [7.5, 6.5]
]), None)

labels = knn_shc(D_L, D_U)
print("Final Cluster Assignments:", labels)

# 使用 breast cancer 数据集进行测试
# data = load_breast_cancer()
# X = data.data
# y = data.target
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # 创建有标签和无标签数据集
# D_L = (X_train[:200], y_train[:200])
# D_U = (X_train[200:], None)
#
# labels = knn_shc(D_L, D_U)
# labels_test = knn_shc((X_train, y_train), (X_test, None))
#
# # 计算指标
# acc = accuracy_score(y_test, labels_test[len(X_train):])
# rn = adjusted_rand_score(y_test, labels_test[len(X_train):])
# nmi = normalized_mutual_info_score(y_test, labels_test[len(X_train):])
#
# print(f"ACC: {acc:.4f}")
# print(f"RN: {rn:.4f}")
# print(f"NMI: {nmi:.4f}")
