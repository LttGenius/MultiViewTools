import numpy as np


def tensor2matrix(tensor, 
                  mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def matrix2tensor(mat, 
                  tensor_size, 
                  mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order='F'), 0, mode)

def fft(ten):
    return np.fft.fftn(ten, axis=0)

def ifft(ten):
    return np.fft.ifftn(ten, axis=0)

def unfold(ten):
    return tensor2matrix(ten, 2).T

def fold(ten, s):
    return matrix2tensor(ten.T, s, 2)

def frontMul(ten1, 
             ten2):
    if ten1.shape[0] != ten2.shape[0]:
        return None
    views = ten1.shape[0]
    n1 = ten1.shape[1]
    m1 = ten1.shape[2]
    n2 = ten2.shape[1]
    m2 = ten2.shape[2]
    m_ten = np.zeros([views, n1, m2])
    for i in range(views):
        m_ten[:, :, i] = np.matmul(ten1[:, :, i], ten2[:, :, i])
    return m_ten

def tProduct(ten1, 
             ten2):
    t_ten1 = fft(ten1)
    t_ten2 = fft(ten2)
    t_ten = frontMul(t_ten1, t_ten2)
    t_ten = ifft(t_ten)
    return t_ten

def kProduct(mat1, 
             mat2):
    return np.kron(mat1, mat2)

def bcirc(ten):
    views = ten.shape[0]
    n = ten.shape[1]
    m = ten.shape[2]
    bcirc_mat = np.zeros(views*n, views*m)
    t = 0
    for i in range(views):
        s = 0
        for j in range(views):
            bcirc_mat[s:s+n, t:t+m] = ten[:, :, (j - i + views)%views]
        t += m
    return bcirc_mat

def bdiag(ten:np.ndarray):
    views = ten.shape[0]
    n = ten.shape[1]
    m = ten.shape[2]
    bdiag_mat = np.zeros(views*n, views*m)
    s = 0
    t = 0
    for i in range(views):
        bdiag_mat[s:s+n, t:t+m]
        s = s + n
        t = t + m
    return bdiag_mat

def Frobenius(ten):
    return np.sum(np.abs(ten)**2)**0.5

def is_Diagonal(mat):
    n, m = mat.shape
    if n!=m: 
        return False
    test = mat.reshape(-1)[:-1].reshape(n-1, m+1)
    return ~np.any(test[:, 1:])

def is_fDiagonal(ten):
    views = ten.shape[0]
    for i in range(views):
        if not is_Diagonal(ten[:, :, i]):
            return False
    return True

def DFTMat(n,
           *arg):
    mat = np.ones([n, n], dtype = complex)
    w = np.exp(complex(0, 2*np.pi/n))
    for i in range(1, n):
        mat[0, i] = w ** i
    for i in range(1, n):
        mat[i, 1:] = mat[i-1, 1:] * w
    return mat

def block_diagonalized(ten,
                       *arg):
    n3, n1, n2 = ten.shape
    f = DFTMat(n3)
    finv = np.conj(f)/n3
    I1 = np.eye(n1)
    I2 = np.eye(n2)
    mat = np.matmul(np.matmul(kProduct(f, I1), bcirc(ten)), kProduct( finv, I2))
    return mat

def tSVD(ten,
         full_matrices=True, 
         compute_uv=True):
    v, n, m = ten.shape
    k = min(n, m)
    if full_matrices:
        tu = np.zeros([v, n, n])
        ts = np.zeros([v, n, m])
        tv = np.zeros([v, m, m])
    else:
        tu = np.zeros([v, n, k])
        ts = np.zeros([v, k, k])
        tv = np.zeros([v, k, m])
    ften = fft(ten)
    for i in range(v):
        u, s, v = np.linalg.svd(ften[:, :, i])
        tu[:, :, i] = u
        ts[:, :, i] = s
        tv[:, :, i] = v
    if compute_uv:
        return ifft(tu), ifft(ts), ifft(tv)
    else:
        return ifft(ts)

def tnn(ten,
        *arg):
    v, n, m = ten.shape
    s = tSVD(ten, compute_uv=False)
    return np.sum(s)


class TensorArray:
    def __init__(self,
                 x: np.ndarray):
        self.x = x
        self.shape = self.x.shape

    def __mul__(self,
                other):
        if isinstance(other, TensorArray):
            assert len(self.shape) == len(other.shape), 'Shape Error!'
            if len(self.shape) == 3:
                res = np.zeros(self.shape[1], other.shape[2])
                v = self.shape[0]
                for i in range(v):
                    res[i, :, :] = np.matmul(self.x[i, :, :], other[i, :, :])
            else:
                res = np.matmul(self.x, other.x)
            return res
        else:
            return other * self.x

    def __xor__(self,
                other):
        return tProduct(self.x, other.x)

