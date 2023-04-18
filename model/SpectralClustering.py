import numpy as np
from base import norm2vec
from kmeans import kmeans


def SpectralClustering(s: np.ndarray,
                       n: int):
    N = s.shape[0]
    max_iter = 1000
    rep = 20
    eps = 1e-6
    DN = np.diag(1 / np.sqrt(np.sum(s,axis=0) + eps))
    LapN = np.eye(N) - np.matmul(DN, np.matmul(DN, s))
    uN,sN,vN = np.linalg.svd(LapN)
    kerN = vN[:, N - n: N]
    for i in range(N):
        kerN[i, :] = kerN[i, :] / (norm2vec(kerN[i, :]) + eps)
    groups = kmeans(kerN, n, max_iter=max_iter, eps=eps, dist='Euclidean', rep=rep)
    return groups
