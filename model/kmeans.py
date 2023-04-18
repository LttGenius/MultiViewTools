import numpy as np
from base import norm2vec
from random import sample


def __random_row(mat: np.ndarray,
                 k: int):
    row_total = mat.shape[0]
    sequence = np.arange(row_total)
    np.random.shuffle(sequence)
    return mat[sequence[0: k], :]


def __euclidean_dis(dataset: np.ndarray,
                    centroids: np.ndarray):
    k = centroids.shape[0]
    n = dataset.shape[0]
    distance = np.zeros(dataset.shape[0], k)
    for i in range(n):
        diff = np.abs(dataset[i, :] - centroids)
        squared_diff = diff ** 2
        squared_dist = np.sum(squared_diff, axis=1)
        distance[i, :] = squared_dist ** 0.5
    return distance


def __manhattan_dis(dataset: np.ndarray,
                   centroids: np.ndarray):
    k = centroids.shape[0]
    n = dataset.shape[0]
    distance = np.zeros(n, k)
    for i in range(n):
        diff = np.abs(dataset[i, :] - centroids)
        distance[i, :] = np.sum(diff, axis=1)
    return distance


def __cosine_dis(dataset: np.ndarray,
                 centroids: np.ndarray):
    k = centroids.shape[0]
    n = dataset.shape[0]
    distance = np.zeros([n, k])
    for i in range(n):
        cos = np.sum(dataset[i, :] * centroids[i, :], axis=1) / \
              ( norm2vec(dataset[i, :]) * norm2vec(centroids[i, :]) )
        distance[i, :] = cos
    return distance


dis = {'Euclidean': __euclidean_dis,
       'Manhattan': __manhattan_dis,
       'Cosine': __cosine_dis}


def __compute_centroids(mat,
                        by):
    con_mat = np.hstack([mat, by])
    centroids = np.zeros([np.max(by), mat.shape[1]])
    for i in by:
        centroids[i, :] = np.mean(con_mat[con_mat[:, -1] == i], axis=0)
    return centroids


def __cluster(dataset: np.ndarray,
              centroids: np.ndarray,
              mode: str = 'Euclidean'):
    cla = dis[mode](dataset, centroids)
    min_dist_indices = np.argmin(cla, axis=1)
    return min_dist_indices


def __classify(dataset: np.ndarray,
               centroids: np.ndarray,
               mode: str = 'Euclidean'):
    min_dist_indices = __cluster(dataset, centroids, mode)
    new_centroids = __compute_centroids(dataset, min_dist_indices)
    changed = new_centroids - centroids
    return new_centroids, changed, min_dist_indices


def kmeans(s: np.ndarray,
           k: int,
           **arguments):
    replicates = arguments['rep']
    best_centroids = np.zeros([s.shape[0], k])
    for i in range(replicates):
        centroids = __random_row(s, k)
        new_centroids, changed, tmp = __classify(s, centroids, arguments['dist'])
        chan = np.sum(np.abs(changed) ** 0.5, axis=1) ** 0.5
        eps = arguments['eps']
        max_iter = arguments['max_iter']
        it = 0
        while np.any(chan > eps) and it < max_iter:
            new_centroids, changed, tmp = __classify(s, centroids, arguments['dist'])
            chan = np.sum(np.abs(changed) ** 0.5, axis=1) ** 0.5
            it += 1
        best_centroids = best_centroids + new_centroids
    best_centroids = best_centroids / replicates
    cluster = __cluster(s, best_centroids, arguments['dist'])
    return best_centroids, cluster




