from base.support import tensor2matrix, matrix2tensor
from base.support import Module
import numpy as np


def norm2vec(vec):
    return np.sum(vec**2)**0.5

def norm1vec(vec):
    return np.sum(np.abs(vec))

def normInfvec(vec):
    return np.max(np.abs(vec))

def normNeInfvec(vec):
    return np.min(np.abs(vec))

def normPvec(vec, p):
    return np.sum(np.abs(vec)**p)**(1/p)

def norm2mat(mat):
    eigenvalue, featurevector = np.linalg.eig(np.matmul(mat.T, mat))
    return np.max(eigenvalue) ** 0.5

def norm1mat(mat):
    return np.max(np.sum(np.abs(mat), axis = 0))

def normInfmat(mat):
    return np.max(np.sum(np.abs(mat), axis = 1))

def normFrobeniusmat(mat):
    return np.sum(np.abs(mat)**2)**0.5

def normNuclearmat(mat):
    Sigma = np.linalg.svd(mat,full_matrices=False, compute_uv=False)
    return np.sum(Sigma)

def norm(tar, model, *arg):
    return norm_compute.compute_value(tar, model, *arg)

class norm_compute(Module):
    @staticmethod
    def compute(
                tar: np.ndarray,
                model: str,
                *arg):
        t = tar.size()
        if t[0] == 1 or t[1] == 1:
            di = {"1": norm1vec,
                  "2": norm2vec,
                  "p": lambda vec: normPvec(vec, *arg),
                  "inf": normInfvec,
                  "-inf": normNeInfvec}
        else:
            di = {"1": norm1mat,
                  "2": norm2mat,
                  "inf": normInfmat,
                  "F": normFrobeniusmat,
                  "*": normNuclearmat}
        return di[model](tar)


def normInfTensor(ten):
    return np.max(np.abs(ten))
