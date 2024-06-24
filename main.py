import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


def calculate_centroids(X, labels, num_clusters):
    centroids = []
    for i in range(num_clusters):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        else:
            centroids.append(np.zeros(X.shape[1]))
    return np.array(centroids)


def assign_clusters(X, centroids, lambda_param):
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
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_labeled)

    pseudo_labels = []
    distances, indices = knn.kneighbors(X_u)

    for i in range(len(X_u)):
        neighbors_indices = indices[i]
        neighbors_labels = y_labeled[neighbors_indices]
        centroid_distances = np.linalg.norm(X_u[i] - centroids, axis=1)
        centroid_label = np.argmin(centroid_distances)

        knn_label = np.argmax(np.bincount(neighbors_labels.astype(int)))

        if np.sum(neighbors_labels == centroid_label) >= k - 2 and centroid_label == knn_label:
            pseudo_labels.append(np.append(X_u[i], centroid_label))

    return np.array(pseudo_labels)


def split_clusters(X, labels, num_clusters):
    new_labels = labels.copy()
    new_clusters_created = False
    for cluster in range(num_clusters):
        cluster_points = X[labels == cluster]
        cluster_labels = labels[labels == cluster]

        unique_labels = np.unique(cluster_labels)
        if len(unique_labels) > 1:
            for unique_label in unique_labels:
                new_cluster_label = max(new_labels) + 1
                new_labels[(labels == cluster) & (cluster_labels == unique_label)] = new_cluster_label
                new_clusters_created = True

    return new_labels, new_clusters_created


def transfer_samples(X, labels, centroids, k=5):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    neighbors = knn.kneighbors(X, return_distance=False)

    new_labels = labels.copy()
    for i in range(len(X)):
        majority_label = np.argmax(np.bincount(labels[neighbors[i]]))
        if np.sum(labels[neighbors[i]] == labels[i]) <= k // 2:
            new_labels[i] = majority_label

    return new_labels


def compute_loss(X_L, y_L, X_U, labels, centroids, lambda_param):
    y_L_pred = labels[:len(X_L)]
    err = np.mean(y_L != y_L_pred)
    sse = np.sum([np.linalg.norm(x - centroids[label]) ** 2 for x, label in zip(np.vstack([X_L, X_U]), labels)])
    return err, sse


def normalize_losses(err_list, sse_list):
    err_max = max(err_list)
    sse_max = max(sse_list)

    err_normalized = [err / err_max for err in err_list]
    sse_normalized = [sse / sse_max for sse in sse_list]

    return err_normalized, sse_normalized


def knn_shc(D_L, D_U, lambda_param=0.5, k=5, max_iter=100):
    X_L = D_L[:, :-1]
    y_L = D_L[:, -1].astype(int)
    X_U = D_U[:, :-1]

    X_P = np.empty((0, X_L.shape[1]))  # 初始化伪标签数据为空
    y_P = np.empty(0)  # 初始化伪标签为空

    num_clusters = len(np.unique(y_L))
    # 更新质心矩阵
    centroids = calculate_centroids(X_L, y_L, num_clusters)

    err_list = []
    sse_list = []

    for iteration in range(max_iter):
        # 步骤2：使用质心和KNN预测无标签数据的簇
        pseudo_labels = compute_pseudo_labels(X_U, X_L, y_L, centroids, k=k)

        if len(pseudo_labels) == 0:
            break

        X_P = np.vstack((X_P, pseudo_labels[:, :-1]))
        y_P = np.hstack((y_P, pseudo_labels[:, -1].astype(int)))

        # 步骤3：用有标签和伪标签数据更新质心
        centroids = calculate_centroids(np.vstack([X_L, X_P]), np.hstack([y_L, y_P]), num_clusters)

        # 步骤4：重新分配簇标签和更新标签
        all_data = np.vstack([X_L, X_U])
        labels = assign_clusters(all_data, centroids, lambda_param)

        # 分裂包含多个类的簇
        labels, new_clusters_created = split_clusters(all_data, labels, num_clusters)

        # 根据KNN多数投票转移样本
        labels = transfer_samples(all_data, labels, centroids, k=k)

        # 计算损失
        err, sse = compute_loss(X_L, y_L, X_U, labels, centroids, lambda_param)
        err_list.append(err)
        sse_list.append(sse)

        if iteration > 0:
            err_normalized, sse_normalized = normalize_losses(err_list, sse_list)
            loss = err_normalized[-1] + sse_normalized[-1]
        else:
            loss = err + sse

        print(f"Iteration {iteration}, Loss: {loss:.4f}")

        if not new_clusters_created or (iteration > 0 and loss >= (err_normalized[-2] + sse_normalized[-2])):
            print("Stopping early due to no new clusters created or increase in loss")
            break

        accuracies = []
        for cluster in range(num_clusters):
            cluster_labels = labels[:len(X_L)][labels[:len(X_L)] == cluster]
            cluster_true_labels = y_L[labels[:len(X_L)] == cluster]
            if len(cluster_true_labels) > 0:
                acc = accuracy_score(cluster_true_labels, cluster_labels)
                accuracies.append(acc)

        mean_accuracy = np.mean(accuracies)

        for cluster in range(num_clusters, max(labels) + 1):
            cluster_labels = labels[:len(X_L)][labels[:len(X_L)] == cluster]
            cluster_true_labels = y_L[labels[:len(X_L)] == cluster]
            if len(cluster_true_labels) > 0:
                acc = accuracy_score(cluster_true_labels, cluster_labels)
                if acc < mean_accuracy:
                    labels[labels == cluster] = -1

    return labels


# 使用 Iris 数据集进行测试
# data = load_iris()
data = load_breast_cancer()
X = data.data
y = data.target

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建有标签和无标签数据集
D_L = np.column_stack((X_train[:200], y_train[:200]))  # 选择前200个样本作为有标签数据
D_U = np.column_stack((X_train[200:], np.full(len(X_train[200:]), -1)))  # 剩余的作为无标签数据

# 运行KNN_SHC算法
final_clusters = knn_shc(D_L, D_U, lambda_param=0.90, k=5, max_iter=100)

# 在测试集上进行预测
test_clusters = knn_shc(np.column_stack((X_train, y_train)), np.column_stack((X_test, np.full(len(X_test), -1))),
                        lambda_param=0.90, k=5, max_iter=100)

# 计算指标
acc = accuracy_score(y_test, test_clusters[len(X_train):])
rn = adjusted_rand_score(y_test, test_clusters[len(X_train):])
nmi = normalized_mutual_info_score(y_test, test_clusters[len(X_train):])

print(f"ACC: {acc:.4f}")
print(f"RN: {rn:.4f}")
print(f"NMI: {nmi:.4f}")
