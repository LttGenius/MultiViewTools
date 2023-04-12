import numpy as np


def tensor2matrix(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def matrix2tensor(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order='F'), 0, mode)


def kProduct(ten1, ten2):

def bcirc(ten):

def bdiag(ten):

def Frobenius(ten):

def is_fDiagonal(ten):

def block_diagonalized(ten):

def tSVD(ten):

def tnn(ten):