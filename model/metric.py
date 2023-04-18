import numpy as np
from Hungarian import Hungarian
import math

def hungarian(x,
              y):
    K = np.unique(y)
    K_num = len(K)
    cost_mat = np.array(np.zeros((K_num, K_num)))
    for i in range(K_num):
        temp_i = K[i]
        idx = np.where(np.array(x) == temp_i)
        for j in range(K_num):
            temp_j = K[j]
            h = y[idx]
            t = np.where(np.array(h) != temp_j)
            cost_mat[i, j] = len(t[0])
    hungarian = Hungarian(input_matrix=cost_mat)
    hungarian.calculate()
    return hungarian.get_results()


def best_map(x,
             y):
    nclass = len(np.unique(y))
    hungarian_res = hungarian(x, y)
    new_x = np.zeros(x.shape)
    for i in range(nclass):
        new_x[np.argwhere(x == i)] = hungarian_res[i][0]
    return new_x

def accuracy(x,
             y,
             *options):
    res = best_map(x, y)
    acc = len(np.argwhere(y == res)) / len(y)
    return acc

def normalized_mutual_information(x,
                                  y,
                                  *options):
    total = len(y)
    A_ids = set(y)
    B_ids = set(x)
    # 互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(y == idA)
            idBOccur = np.where(x == idB)
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            px = 1.0 * len(idAOccur[0]) / total
            py = 1.0 * len(idBOccur[0]) / total
            pxy = 1.0 * len(idABOccur) / total
            MI = MI + pxy * math.log(pxy / (px * py) + eps, 2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0 * len(np.where(y == idA)[0])
        Hx = Hx - (idAOccurCount / total) * math.log(idAOccurCount / total + eps, 2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0 * len(np.where(x == idB)[0])
        Hy = Hy - (idBOccurCount / total) * math.log(idBOccurCount / total + eps, 2)
    MIhat = 2.0 * MI / (Hx + Hy)
    return MIhat

class compute_p_r_f:
    p = 0
    r = 0
    f = 0
    ri = -1
    def __init__(self,
                 x,
                 y):
        self.x = x
        self.y = y
        self.com = False
    def compute(self):
        x = self.x
        y = self.y
        if len(x) != len(y):
            return
        N = len(x)
        numT = 0
        numH = 0
        numI = 0
        for n in range(N):
            Tn = np.zeros(x.shape)
            Tn[np.argwhere(x[n:] == x(n))] = 1

            Hn = np.zeros(x.shape)
            Hn[np.argwhere(y[n:] == y(n))] = 1
            numT = numT + np.sum(Tn)
            numH = numH + np.sum(Hn)
            numI = numI + np.sum(Tn * Hn)
        p = 1
        r = 1
        f = 1
        if numH > 0:
            p = numI / numH
        if numT > 0:
            r = numI / numT
        if (p + r) == 0:
            f = 0
        else:
            f = 2 * p * r / (p + r)
        self.p = p
        self.r = r
        self.f = f
        self.com = True
        return p, r, f

def recall(x,
           y,
           *options):
    if options:
        com_class = options[0]
        if com_class.com:
            return com_class.r
        else:
            com_class.compute()
            return com_class.r
    return None

def f_score(x,
           y,
            *options):
    if options:
        com_class = options[0]
        if com_class.com:
            return com_class.f
        else:
            com_class.compute()
            return com_class.f
    return None

def precision(x,
              y,
              *options):
    if options:
        com_class = options[0]
        if com_class.com:
            return com_class.p
        else:
            com_class.compute()
            return com_class.p
    return None

def adjust_random_index(x,
                        y,
                        *options):
    com_cls = options[0]
    if com_cls.ri > -1:
        eri = np.mean(com_cls.ri)
        ri = com_cls.ri
        return (ri - eri) / (np.max(ri) - eri)
    else:
        ri = random_index(x, y)
        eri = np.mean(ri)
        return (ri - eri) / (np.max(ri) - eri)


def random_index(x,
                 y,
                 *options):
    if options[0].ri > -1:
        return options[0].ri
    else:
        y = np.concatenate([y, np.clip(1 - np.sum(y, axis=0, keepdims=True), a_min=0, a_max=1)], axis=0)
        x = np.concatenate([x, np.clip(1 - np.sum(x, axis=0, keepdims=True), a_min=0, a_max=1)],
                                   axis=0)
        T = (np.expand_dims(y, axis=1) * x).sum(-1).sum(-1).astype(np.float32)
        # The contingency table
        N = T.sum()
        RI = 1 - ((np.power(T.sum(0), 2).sum() + np.power(T.sum(1), 2).sum()) / 2 - np.power(T, 2).sum()) / (
                N * (N - 1) / 2)
        options[0].ri = RI
        return RI


def purity(x,
           y):
    y_labels = np.zeros(y.shape)
    labels = np.unique(y)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y[y == labels[k]] = ordered_labels[k]
    labels = np.unique(y)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)
    for cluster in np.unique(x):
        hist, _ = np.histogram(y[x == cluster], bins=bins)
        winner = np.argmax(hist)
        y_labels[x == cluster] = winner
    return accuracy(y, y_labels)


__metric_dict = {
    'purity': purity,
    'acc': accuracy,
    'nmi': normalized_mutual_information,
    'ARI': adjust_random_index,
    'RI': random_index,
    'precision': precision,
    'f-score': f_score,
    'recall': recall
}


def metric(x: np.ndarray,
           y: np.ndarray,
           *options):
    n = max(y.shape)
    uy = np.unique(y)
    nclass = max(uy.shape)
    y0 = np.zeros([n, 1])
    if nclass != max(uy):
        for i in range(1, nclass+1):
            y0[np.argwhere(y == uy[i])] = i
        y = y0

    uy = np.unique(x)
    nclass = max(uy.shape)
    predy0 = np.zeros([n, 1])
    if nclass != max(uy):
        for i in range(1, nclass+1):
            predy0[np.argwhere(x == uy[i])] = i
        x = predy0
    re_me = []
    if not options:
        options = __metric_dict.keys()
    com_cls = compute_p_r_f(x, y)
    for i in options:
        re_me.append(__metric_dict[i](x, y, com_cls))
    return re_me

