from base.tensor import TNN
from base import normInfmat
from base.tensor import normInfTensor
from base import Admm
from base import Spare21Norm
from base import Convergence
from base import Module
from base import Connect
from model import base_model
from model import SpectralClustering
from model import metric
import numpy as np


class c1(Connect):
    def optimization(self,
                     arg=None):
        v = self.variables['X'].shape[0]
        res = np.zeros(self.variables['Z'].shape)
        for i in range(v):
            tmp = self.variables['X'][i, :, :] - \
                  self.variables['E'][i, :, :] + \
                  self.variables['Y'] / self.variables['mu']
            res[i, :, :] = tmp
        return res


class c2(Connect):
    def optimization(self,
                     arg=None):
        v = self.variables['X'].shape[0]
        res = np.zeros(self.variables['Z'].shape)
        for i in range(v):
            tmp = self.variables['X'][i, :, :] - \
                  self.variables['Z'][i, :, :] + \
                  self.variables['Y'] / self.variables[
                'mu']
            res[i, :, :] = tmp
        return res


class c3(Connect):
    def optimization(self,
                     arg=None):
        v = self.variables['X'].shape[0]
        res = np.zeros(self.variables['Z'].shape)
        for i in range(v):
            tmp = self.variables['X'][i, :, :] - \
                  self.variables['Z'][i, :, :] - self.variables['E']
            res[i, :, :] = tmp
        return res



class m1(Module):
    def optimization(self,
                     arg=None):
        v = self.variables[self.opt_variable].shape[0]
        for i in range(v):
            self.variables[self.opt_variable][i, :, :] = \
                self.variables[self.opt_variable][i, :, :] + \
                self.variables[self.arguments[1]] * arg[i, :, :]


class conv(Convergence):
    def __init__(self,
                 *arg):
        super().__init__(*arg)
        self.lastZ = 0
        self.lastE = 0

    def optimization(self,
                     arg=None):
        var = self.variables['X'] - \
              self.variables['Z'] - \
              self.variables['E']
        c1 = normInfTensor(
                        self.lastZ -
                        self.variables['Z']
                        )
        c2 = normInfTensor(
                        self.lastE -
                        self.variables['E']
                        )
        c3 = normInfTensor(
                        var
                        )
        if c1 < arg \
                and \
                c2 < arg \
                and \
                c3 < arg:
            t1 = True
        else:
            t1 = False
        return c1, \
            c2, \
            c3, \
            t1


class etlmsc(base_model):
    def load_mm(self):
        sx = self.X.shape
        V = sx[0]
        M = sx[1]
        N = sx[2]
        variables = {'X': self.X,
                     'label': self.Y,
                     'Z': np.zeros([V, N, N]),
                     'E': np.zeros([V, M, N]),
                     'Y': np.zeros([V, M, N]),
                     'mu': 10e-3,
                     'max_mu': 10e10,
                     'pho_mu': 2,
                     'N': N}
        variables = dict(**variables, **self.arg)
        arguments = {'max_iter': 200,
                     'eps': 1e-7}
        self.opt_method = Admm(variable=variables,
                               arguments=arguments)
        self.opt_method \
        << \
        c1() \
        << \
        TNN('Z', 'mu') \
        << \
        c2() \
        << \
        Spare21Norm('E', 'lambda', 'mu') \
        << \
        c3() \
        << \
        m1('Y', 'mu') \
        >> \
        conv()


def run_etlmsc(x: np.ndarray,
                n: int,
                y: np.ndarray = None,
                arguments: dict = None):
    if not arguments:
        arguments['lambda'] = 0.5
    model_tsvdmsc = etlmsc(
        x=x,
        y=y,
        argument=arguments
    )
    var = model_tsvdmsc.run()
    z = var['Z']
    v = z.shape[0]
    s = 0
    for i in range(v):
        tmp = z[i, :, :]
        s = s + (np.abs(tmp) + np.abs(tmp).T) / 2
    pre_y = SpectralClustering(
        s,
        n
    )
    me = metric(
        pre_y,
        y
    )
    return pre_y, me




