import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
import pandas as pd


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


def compute_pseudo_labels(X_u, X_labeled, y_labeled, centroids, k=5):
    if len(X_labeled) == 0 or len(X_u) == 0:
        return np.empty((0, X_u.shape[1] + 1))

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


def assign_clusters(X, centroids, lambda_param=0.5, k=5):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    D = np.sum(distances, axis=1)
    weighted_distances = distances * lambda_param
    labels = np.argmin(weighted_distances, axis=1)

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    _, indices = knn.kneighbors(X)

    for i in range(len(X)):
        nearest_labels = labels[indices[i]]
        majority_label = np.argmax(np.bincount(nearest_labels))
        centroid_distances = np.linalg.norm(X[i] - centroids, axis=1)
        nearest_centroid = np.argmin(centroid_distances)

        if np.sum(nearest_labels == nearest_centroid) >= k - 2:
            labels[i] = nearest_centroid

    return labels


def split_clusters(X, labels, num_clusters):
    new_labels = labels.copy()
    new_clusters_created = False
    clusters_to_check = list(range(num_clusters))  # 初始时需要检查的簇列表

    while clusters_to_check:
        cluster = clusters_to_check.pop(0)
        cluster_points = X[labels == cluster]
        cluster_labels = labels[labels == cluster]  # 获取当前簇内的标签

        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        if len(unique_labels) > 1:
            for unique_label in unique_labels:
                if unique_label == cluster:
                    continue

                new_cluster_label = np.max(new_labels) + 1
                new_labels[(labels == cluster) & (cluster_labels == unique_label)] = new_cluster_label
                new_clusters_created = True

                # 添加新创建的簇标签到待检查列表中
                clusters_to_check.append(new_cluster_label)

    return new_labels, new_clusters_created


def transfer_samples(X, labels, centroids, k=5):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)

    for i in range(len(X)):
        distances, indices = knn.kneighbors(X[i].reshape(1, -1))
        nearest_labels = labels[indices[0]]
        majority_label = np.argmax(np.bincount(nearest_labels))
        centroid_distances = np.linalg.norm(X[i] - centroids, axis=1)
        nearest_centroid = np.argmin(centroid_distances)

        if majority_label != nearest_centroid:
            labels[i] = nearest_centroid

    return labels


def compute_loss(X_L, y_L, X_U, labels, centroids, lambda_param=0.5):
    labeled_distances = np.linalg.norm(X_L - centroids[y_L], axis=1)
    labeled_loss = np.sum(labeled_distances)

    unlabeled_distances = np.linalg.norm(X_U - centroids[labels[len(X_L):]], axis=1)
    unlabeled_loss = np.sum(unlabeled_distances)

    total_loss = labeled_loss + lambda_param * unlabeled_loss

    sse = np.sum((X_L - centroids[y_L]) ** 2) + np.sum((X_U - centroids[labels[len(X_L):]]) ** 2)
    return total_loss, sse


def normalize_losses(err_list, sse_list):
    if np.max(err_list) - np.min(err_list) == 0 or np.max(sse_list) - np.min(sse_list) == 0:
        return np.zeros_like(err_list), np.zeros_like(sse_list)

    err_normalized = (err_list - np.min(err_list)) / (np.max(err_list) - np.min(err_list))
    sse_normalized = (sse_list - np.min(sse_list)) / (np.max(sse_list) - np.min(sse_list))
    return err_normalized, sse_normalized


def knn_shc(D_L, D_U, X_test, lambda_param=0.5, k=5, tol=1e-4, max_iter=100):
    X_L = D_L[:, :-1]
    y_L = D_L[:, -1].astype(int)
    X_U = D_U

    X_P = np.empty((0, X_L.shape[1]))  # 初始化伪标签数据为空
    y_P = np.empty(0)  # 初始化伪标签为空

    num_clusters = len(np.unique(y_L))
    # 更新质心矩阵
    centroids = calculate_centroids(X_L, y_L, num_clusters)

    err_list = []
    sse_list = []

    labels = None
    prev_loss = None
    iteration = 0

    while iteration < max_iter:
        print(f"迭代次数Iteration {iteration}")
        # 步骤2：使用质心和KNN预测无标签数据的簇
        if len(X_U) == 0:
            break

        pseudo_labels = compute_pseudo_labels(X_U, X_L, y_L, centroids, k=k)
        print(f"产生 {len(pseudo_labels)} 伪标签.")

        if len(pseudo_labels) == 0:
            break

        X_P = np.vstack((X_P, pseudo_labels[:, :-1]))
        y_P = np.hstack((y_P, pseudo_labels[:, -1].astype(int)))

        # 步骤3：用有标签和伪标签数据更新质心
        centroids = calculate_centroids(np.vstack([X_L, X_P]), np.hstack([y_L, y_P]), num_clusters)

        # 步骤4：重新分配簇标签和更新标签
        all_data = np.vstack([X_L, X_U])
        labels = assign_clusters(all_data, centroids, lambda_param, k)
        print(f"标签分配结果：{np.unique(labels, return_counts=True)}")

        # 确保没有空簇
        unique_labels, counts = np.unique(labels, return_counts=True)
        empty_clusters = np.setdiff1d(np.arange(num_clusters), unique_labels)
        if len(empty_clusters) > 0:
            print(f"警告：出现空簇 {empty_clusters}")

        # 分裂包含多个类的簇
        labels, new_clusters_created = split_clusters(all_data, labels, num_clusters)
        if new_clusters_created:
            num_clusters = max(labels) + 1  # 更新簇数量
            print("产生新簇.")
        else:
            print("无新簇产生.")

        # 根据KNN多数投票转移样本
        labels = transfer_samples(all_data, labels, centroids, k=k)

        # 计算损失
        err, sse = compute_loss(X_L, y_L, X_U, labels, centroids, lambda_param)
        err_list.append(err)
        sse_list.append(sse)

        err_normalized, sse_normalized = normalize_losses(err_list, sse_list)
        loss = err_normalized[-1] + sse_normalized[-1]

        print(f"损失函数Loss: {loss:.4f}")

        if prev_loss is not None and abs(prev_loss - loss) < tol:
            print("停止迭代，损失函数增大")
            break

        prev_loss = loss
        iteration += 1

    return labels


def run_experiment(dataset_loader, dataset_name, lambda_param, k, tol, random_state, max_iter):
    # 加载数据集
    data = dataset_loader()
    X = data.data
    y = data.target

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # 使用一部分数据作为有标签数据
    labeled_indices = np.random.choice(len(X_train), min(200, len(X_train)), replace=False)
    X_labeled = X_train[labeled_indices]
    y_labeled = y_train[labeled_indices]

    # 剩余部分作为无标签数据
    unlabeled_indices = np.setdiff1d(np.arange(len(X_train)), labeled_indices)
    X_unlabeled = X_train[unlabeled_indices]

    # 准备有标签和无标签数据集
    D_L = np.hstack([X_labeled, y_labeled.reshape(-1, 1)])
    D_U = X_unlabeled

    # 运行KNN-SHC算法
    all_labels = knn_shc(D_L, D_U, X_test, lambda_param=lambda_param, k=k, tol=tol, max_iter=max_iter)

    # 评估结果
    test_clusters = all_labels[len(X_train):]
    acc = accuracy_score(y_test, test_clusters)
    ari = adjusted_rand_score(y_test, test_clusters)
    nmi = normalized_mutual_info_score(y_test, test_clusters)

    results = {
        "dataset": dataset_name,
        "accuracy": acc,
        "adjusted_rand_index": ari,
        "normalized_mutual_info_score": nmi
    }

    return results


# 数据集名称列表
dataset_names = [
    "load_breast_cancer", "load_iris", "load_wine", "load_digits"
]

all_results = []

# 逐个运行实验
for dataset_name in dataset_names:
    dataset_loader = getattr(datasets, dataset_name)
    result = run_experiment(dataset_loader, dataset_name, lambda_param=0.90, k=5, tol=1e-4, random_state=None,
                            max_iter=100)
    all_results.append(result)

# 将结果保存到CSV文件中
results_df = pd.DataFrame(all_results)
results_df.to_csv("knn_shc_results.csv", index=False)

print("Results saved to knn_shc_results.csv")
