import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score


class KNN_SHC_Clusterer:
    def __init__(self, lambda_param=0.5, k=5, max_iter=10):
        self.lambda_param = lambda_param
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def fit(self, X_L, y_L, X_U):
        num_clusters = len(np.unique(y_L))
        X_LU = np.vstack([X_L, X_U])
        self.centroids = self.calculate_centroids(X_L, y_L, num_clusters)
        self.labels = np.zeros(X_LU.shape[0], dtype=int)
        self.labels[:len(X_L)] = y_L

        print(f"Initial centroids:\n{self.centroids}")

        iteration = 0
        prev_loss = np.inf

        while iteration < self.max_iter:
            print(f"Iteration {iteration + 1}")
            iteration += 1

            pseudo_labels = self.compute_pseudo_labels(X_U, X_L, y_L)
            if len(pseudo_labels) == 0:
                print("No more pseudo-labels could be assigned.")
                break

            X_P, y_P = pseudo_labels[:, :-1], pseudo_labels[:, -1]
            combined_X = np.vstack([X_L, X_P])
            combined_y = np.hstack([y_L, y_P])

            self.centroids = self.calculate_centroids(combined_X, combined_y, num_clusters)
            print(f"Debug: Updated centroids:\n{self.centroids}")

            self.labels = self.assign_clusters_probabilistically(X_LU, self.centroids, num_clusters)
            print(f"Debug: X_LU shape {X_LU.shape}, self.labels length {len(self.labels)}")
            print(f"Debug: After assign_clusters_probabilistically, self.labels length {len(self.labels)}")

            num_clusters = self.split_clusters_with_multiple_classes(combined_X, combined_y, num_clusters, X_LU)

            self.reassign_based_on_neighbors(X_LU, self.centroids)

            error = self.compute_error(X_L, y_L, self.labels[:len(X_L)])
            sse = self.compute_sse(X_LU, self.labels, self.centroids)

            total_loss = self.compute_total_loss(error, sse)
            print(f'Current loss: {total_loss}')

            if total_loss >= prev_loss:
                print("Loss stabilization detected; stopping iteration.")
                break
            prev_loss = total_loss

        return self

    def calculate_centroids(self, X, y, num_clusters):
        centroids = []
        for i in range(num_clusters):
            cluster_points = X[y == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
            else:
                centroid = np.random.rand(X.shape[1])  # Avoid empty clusters
            centroids.append(centroid)
        return np.array(centroids)

    def compute_pseudo_labels(self, X_U, X_L, y_L):
        knn = NearestNeighbors(n_neighbors=self.k)
        knn.fit(X_L)
        pseudo_labels = []

        for x_u in X_U:
            distances, indices = knn.kneighbors([x_u])
            neighbor_labels = y_L[indices[0]]
            most_common_label = np.bincount(neighbor_labels).argmax()

            if np.bincount(neighbor_labels)[most_common_label] > self.k // 2:
                dists_to_centroids = np.linalg.norm(x_u - self.centroids, axis=1)
                predicted_centroid_label = np.argmin(dists_to_centroids)

                if most_common_label == predicted_centroid_label:
                    pseudo_labels.append(np.append(x_u, most_common_label))

        print(f"Debug: Generated {len(pseudo_labels)} pseudo-labels")
        return np.array(pseudo_labels)

    def assign_clusters_probabilistically(self, X, centroids, num_clusters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        D = np.sum(distances, axis=1)
        labels = np.zeros(X.shape[0], dtype=int)

        for i, x in enumerate(X):
            probabilities = np.zeros(num_clusters)
            for j in range(num_clusters):
                dist_metric = (D[i] - distances[i, j]) / D[i]
                neighbor_metric = np.sum(self.labels == j) / len(self.labels)
                probabilities[j] = self.lambda_param * dist_metric + (1 - self.lambda_param) * neighbor_metric
            labels[i] = np.argmax(probabilities)

        return labels

    def split_clusters_with_multiple_classes(self, X, labels, num_clusters):
        unique_labels = np.unique(labels)
        new_clusters = []
        print(f"Debug: Initial number of clusters: {num_clusters}")

        for cluster in unique_labels:
            cluster_indices = np.where(labels == cluster)[0]
            cluster_labels = labels[cluster_indices]
            true_labels = np.unique(cluster_labels)
            print(f"Debug: Checking cluster {cluster}, size {len(cluster_indices)}")
            print(f"Debug: Cluster {cluster} true labels: {np.unique(true_labels, return_counts=True)}")

            if len(true_labels) > 1:
                for label in true_labels:
                    label_indices = cluster_indices[cluster_labels == label]
                    if len(label_indices) > 0:
                        print(f"Debug: Splitting cluster {cluster} for label {label}, size {len(label_indices)}")
                        new_clusters.append((label, X[label_indices]))

        if new_clusters:
            new_centroids = []
            for label, points in new_clusters:
                new_centroids.append(np.mean(points, axis=0))
            self.centroids = np.vstack([self.centroids, new_centroids])
            num_clusters += len(new_clusters)
            print(f"Debug: New centroids added, total clusters now: {num_clusters}")

        self.labels = self.assign_clusters_probabilistically(X, self.centroids, num_clusters)
        print(
            f"Debug: After split_clusters_with_multiple_classes, self.labels length {len(self.labels)}, num_clusters {num_clusters}")
        return num_clusters

    def reassign_based_on_neighbors(self, X, centroids):
        knn = NearestNeighbors(n_neighbors=self.k)
        knn.fit(X)
        for i, x in enumerate(X):
            distances, indices = knn.kneighbors([x])
            if np.any(indices[0] >= len(self.labels)):
                print(f"Debug: Skipping reassignment for index {i} due to out-of-bounds indices.")
                continue
            neighbor_labels = self.labels[indices[0]]
            most_common_label = np.bincount(neighbor_labels).argmax()
            if np.bincount(neighbor_labels)[most_common_label] > self.k // 2 and most_common_label != self.labels[i]:
                self.labels[i] = most_common_label

    def compute_error(self, X_L, y_L, labels):
        error_rate = np.mean(y_L != labels)
        print(f"Debug: Error rate {error_rate}")
        return error_rate

    def compute_sse(self, X, labels, centroids):
        sse = 0
        for i, centroid in enumerate(centroids):
            cluster_points = X[labels == i]
            sse += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
        print(f"Debug: SSE {sse}")
        return sse

    def compute_total_loss(self, error, sse):
        total_loss = self.lambda_param * error + (1 - self.lambda_param) * sse
        print(f"Debug: Total loss {total_loss}")
        return total_loss


def run_experiment_blobs():
    X, y = make_blobs(n_samples=1000, centers=5, cluster_std=1.0, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_L, X_U, y_L, _ = train_test_split(X_train, y_train, test_size=0.75, random_state=42)

    clusterer = KNN_SHC_Clusterer(lambda_param=0.90, k=5, max_iter=10)
    clusterer.fit(X_L, y_L, X_U)

    combined_data = np.vstack([X_L, X_test])
    all_labels = clusterer.assign_clusters_probabilistically(combined_data, clusterer.centroids, len(np.unique(y_L)))
    y_pred = all_labels[len(X_L):]

    ari = adjusted_rand_score(y_test, y_pred)
    nmi = normalized_mutual_info_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Adjusted Rand Index:", ari)
    print("Normalized Mutual Information Score:", nmi)
    print("Accuracy:", acc)


if __name__ == "__main__":
    run_experiment_blobs()
