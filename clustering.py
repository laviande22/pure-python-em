"""Pure Python implementation of EM algorithm."""
from array import array
import random


class Cluster:
    """Implementation of EM clustering."""
    def __init__(self, filename, dim, num_entry, num_cluster=10):
        self.float_size = 4
        self.dim = dim
        self.num_entry = num_entry
        self.data = self.import_data(filename)
        self.num_cluster = num_cluster
        self.centroids = random.sample(self.data, self.num_cluster)
        self.labels = {cluster_idx: [] for cluster_idx in range(self.num_cluster)}

    def import_data(self, filename):
        """Read and process the binary data."""
        raw_data = array('f')
        with open(filename, 'rb') as file_desc:
            raw_data.frombytes(file_desc.read())
        data = [[] for _ in range(self.num_entry)]
        for i in range(self.num_entry):
            for j in range(self.dim):
                idx = i * self.dim + j
                data[i].append(raw_data[idx])
        return data

    def distance(self, lhs, rhs):
        """Euclidean distance between two vectors 'lhs' and 'rhs'."""
        return sum([(lhs[idx] - rhs[idx]) ** 2 for idx in range(self.dim)]) ** 0.5

    def vectors_mean(self, vectors):
        """Calculate the mean of a set of vectors."""
        total = [0 for _ in range(self.dim)]
        for vector in vectors:
            total = [lhs + rhs for lhs, rhs in zip(total, vector)]
        return [elem / len(vectors) for elem in total]

    def e_step(self):
        """E-step in EM algorithm: Expectation."""
        self.labels = {cluster_idx: [] for cluster_idx in range(self.num_cluster)}
        for vector_idx, vector in enumerate(self.data):
            distances_to_clusters = []
            for cluster in self.centroids:
                distances_to_clusters.append(self.distance(vector, cluster))
            min_cluster = distances_to_clusters.index(min(distances_to_clusters))
            self.labels[min_cluster].append(vector_idx)

    def m_step(self):
        """M-step in EM algorithm: Maximization."""
        new_centroids = []
        for _, member_indices in self.labels.items():
            member_vectors = [self.data[idx] for idx in member_indices]
            new_centroids.append(self.vectors_mean(member_vectors))
        loss = sum([self.distance(new, old)
                    for new, old
                    in zip(new_centroids, self.centroids)]) / self.num_cluster
        self.centroids = new_centroids
        return loss

    def fit(self, epsilon=0.001, max_iter=200):
        """Fitting to data."""
        for idx in range(max_iter):
            self.e_step()
            loss = self.m_step()
            print('step {}: loss {}.'.format(idx, loss))
            if loss < epsilon:
                for member_indices in self.labels.items():
                    print(member_indices)
                for cluster_idx, centroid in enumerate(self.centroids):
                    print('cluster {}: {}'.format(cluster_idx, centroid))
                break

    def cluster_vectors(self, cluster):
        """Return vectors of a cluster."""
        return [self.data[idx] for idx in self.labels[cluster]]
