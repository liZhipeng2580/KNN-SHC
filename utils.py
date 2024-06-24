
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import NearestNeighbors


# 加载数据集
def loadfile(filename):
    X_train, y_train = load_svmlight_file(filename)
    x = X_train.todense()
    y_train = y_train.astype(int)
    return np.array(x), np.array(y_train)


# 计算两个点的欧式距离
def distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))
    # return torch.sqrt(torch.sum(torch.square(x - y)))

# KNN_SHC算法
def knn_shc(D_L, D_U, D_P, lambda_value, n_clusters, n_neighbors=5):
    # 初始化步骤：将有标签数据和伪标签数据合并为X_L∪X_P
    X_L = D_L[:, :-1]  # 有标签数据特征
    y_L = D_L[:, -1]  # 有标签数据标签
    X_U = D_U[:, :-1]  # 无标签数据特征

    X_L_P = np.vstack((X_L, D_P[:, :-1]))  # 所有有标签和伪标签数据的特征
    y_L_P = np.hstack((y_L, D_P[:, -1]))  # 所有有标签和伪标签数据的标签

    # 步骤1：初始聚类
    kmeans_initial = KMeans(n_clusters=n_clusters, random_state=42)
    clusters_initial = kmeans_initial.fit_predict(X_L_P)
    cluster_centers = kmeans_initial.cluster_centers_

    # 步骤2：伪标签生成和加入D_P
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X_L_P)
    distances, indices = knn.kneighbors(X_U)

    for i in range(len(X_U)):
        neighbors_indices = indices[i]
        neighbors_labels = y_L_P[neighbors_indices]
        predicted_cluster = clusters_initial[neighbors_indices[0]]
        # 分别根据簇质心和KNN（K = 5，最近的5个有标签和伪标签样本）预测所属的簇
        # 如果二者一致预测x属于簇〖 p〗_i, 且x近邻中属于簇〖 p〗_i的数量大于3
        # 则将（x, s_i）作为伪标签样本加入D_P
        if np.sum(neighbors_labels == y_L_P[predicted_cluster]) > 3 and clusters_initial[i] == predicted_cluster:
            D_P = np.append(D_P, np.array([np.append(X_U[i], y_L_P[predicted_cluster])]), axis=0)

    # 步骤3：更新质心矩阵M
    X_L_P = np.vstack((X_L, D_P[:, :-1]))
    kmeans_updated = KMeans(n_clusters=n_clusters, random_state=42)
    clusters_updated = kmeans_updated.fit_predict(X_L_P)
    cluster_centers_updated = kmeans_updated.cluster_centers_

    # 步骤4：迭代直到收敛
    while True:
        # 重新分簇
        for h in range(n_clusters):
            cluster_samples = X_L_P[clusters_updated == h]
            if len(np.unique(y_L_P[clusters_updated == h])) > 1:
                for j in np.unique(y_L_P[clusters_updated == h]):
                    new_cluster = cluster_samples[y_L_P[clusters_updated == h] == j]
                    new_center = np.mean(new_cluster, axis=0)
                    cluster_centers_updated = np.append(cluster_centers_updated, [new_center], axis=0)

        # 计算准确率
        accuracies = []
        for h in range(n_clusters):
            cluster_samples = X_L_P[clusters_updated == h]
            cluster_labels = y_L_P[clusters_updated == h]
            if len(cluster_samples) > 0:
                predicted_labels = np.full_like(cluster_labels, cluster_labels[0])
                accuracies.append(np.mean(predicted_labels == cluster_labels))

        # 判断是否收敛
        if new_cluster_satisfies_condition:
            break

    # 步骤5：最终簇划分
    final_clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_L_P)

    return final_clusters

#计算样本x属于某个簇的概率
#x代表样本x，centroids为其他簇的质心，cluster_labels 标签，k_neighbors 邻居数，lambda_param权重参数
def calculate_probability(x, centroids, cluster_labels, k_neighbors, lambda_param):
    # 计算总距离D
    distances = np.linalg.norm(x - centroids, axis=1)
    D = np.sum(distances)

    probabilities = []
    for i, m_i in enumerate(centroids):
        distance_to_cluster = np.linalg.norm(x - m_i)
        D_minus_distance = D - distance_to_cluster
        distance_term = lambda_param * (D_minus_distance / D)

        # 计算近邻情况的信任度
        N_i = np.sum(cluster_labels[:k_neighbors] == i)
        K = k_neighbors
        neighbor_term = (1 - lambda_param) * (N_i / K)

        probability = distance_term + neighbor_term
        probabilities.append(probability)

    return probabilities

# 损失函数
def calculate_loss(X_L, X_U, Y_L, Y_L_pred, centroids, lambda_param):
    # 计算经验损失
    def error(Y_L, Y_L_pred):
        return np.mean(Y_L != Y_L_pred)

    # 计算SSE
    def sse(X, centroids, labels):
        total_sse = 0
        for i, centroid in enumerate(centroids):
            cluster_points = X[labels == i]
            total_sse += np.sum((cluster_points - centroid) ** 2)
        return total_sse

    # 计算Err和SSE
    err = error(Y_L, Y_L_pred)
    all_X = np.vstack((X_L, X_U))
    all_labels = np.hstack((Y_L, np.full(X_U.shape[0], -1)))  # 未标记数据的标签用-1表示
    sse_val = sse(all_X, centroids, all_labels)

    # 计算损失
    loss = lambda_param * err + (1 - lambda_param) * sse_val
    return loss


