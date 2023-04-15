from base.tensor import TNN
from base import normInfmat
from base import Admm
from base import Spare21Norm
from base import Module
from base import Connect
import numpy as np


class c1(Connect):
    def optimization(self,
                     arg=None):
        v = self.variables['X'].shape[0]
        res = np.zeros(self.variables['G'].shape)
        for i in range(v):
            tmp = (np.matmul(self.variables['X'][i, :, :].T, self.variables['Y'][i, :, :]) +
                  self.variables['mu'] * np.matmul(self.variables['X'][i, :, :].T, self.variables['X'][i, :, :]) -
                  self.variables['mu'] * np.matmul(self.variables['X'][i, :, :].T, self.variables['E'][i, :, :]) -
                  self.variables['W'][i, :, :]) / self.variables['rho'] + self.variables['G'][i, :, :]
            res[i, :, :] = tmp
        return res


class c2(Connect):
    def optimization(self,
                     arg=None):
        v = self.variables['X'].shape[0]
        res = np.zeros(self.variables['Y'].shape)
        for i in range(v):
            tmp = self.variables['X'][i, :, :] - np.matmul(self.variables['X'][i, :, :], self.variables['Z'][i, :, :]) + \
                  self.variables['Y'][i, :, :] / self.variables['mu']
            res[i, :, :] = tmp
        return res


class c3(Connect):
    def optimization(self,
                     arg=None):
        v = self.variables['X'].shape[0]
        res = np.zeros(self.variables['Y'].shape)
        for i in range(v):
            tmp = self.variables['X'][i, :, :] - np.matmul(self.variables['X'][i, :, :], self.variables['Z'][i, :, :]) - \
                  self.variables['E'][i, :, :]
            res[i, :, :] = tmp
        return res




class c4(Connect):
    def optimization(self,
                     arg=None):
        res = self.variables['Z'] + self.variables['W'] / self.variables['rho']
        return res

class c5(Connect):
    def optimization(self,
                     arg=None):
        return self.variables['Z'] - self.variables['G']

class m1(Module):
    def optimization(self,
                     arg=None):
        v = self.variables[self.opt_variable].shape[0]
        for i in range(v):
            self.variables[self.opt_variable][i, :, :] \
                = np.matmul(np.linalg.pinv(np.eye(self.variables[self.arguments[0]]) + \
                                 self.variables[self.arguments[1]]/self.variables[self.arguments[2]] * \
                                 np.matmul(self.variables[self.arguments[3]][i, :, :], self.variables[self.arguments[3]][i, :, :].T)), arg[i, :, :])


class m2(Module):
    def optimization(self,
                     arg=None):
        v = self.variables[self.opt_variable].shape[0]
        for i in range(v):
            self.variables[self.opt_variable][i, :, :] = self.variables[self.opt_variable][i, :, :] + \
                                                         self.variables[self.arguments[1]] * arg[i, :, :]


class m3(Module):
    def optimization(self,
                     arg=None):
        self.variables[self.opt_variable] = min(self.variables[self.opt_variable] * self.variables[self.arguments[0]],
                                                self.variables[self.arguments[1]])

class tSVDMSC:
    X = 0
    Y = 0
    mm = []
    arg = {}
    opt_method = None

    def __init__(self,
                 x: np.ndarray = None,
                 y: np.ndarray = None,
                 argument = None):
        self.load(x, y)
        if argument:
            self.load_arg(**argument)

    def load_arg(self,
                 **lam):
        self.arg = lam

    def __load_mm(self):
        sx = self.X.shape
        V = sx[0]
        M = sx[1]
        N = sx[2]
        variables = {'X': self.X,
                     'label': self.Y,
                     'Z': np.zeros([V, N, N]),
                     'W': np.zeros([V, N, N]),
                     'E': np.zeros([V, M, N]),
                     'Y': np.zeros([V, M, N]),
                     'F': np.zeros([V, M, N]),
                     'mu': 10e-3,
                     'max_mu': 10e10,
                     'pho_mu': 2,
                     'rho': 10e-3,
                     'max_rho': 10e10,
                     'pho_rho': 2,
                     'N': N}
        variables = dict(**variables, **self.arg)
        mm = [
            c1(),
            m1('Z', 'N', 'mu', 'rho'),
            c2(),
            Spare21Norm('E', 'lambda', 'mu'),
            c3(),
            m2('Y', 'mu'),
            c4(),
            TNN('G', 'rho'),
            c5(),
            m2('W', 'rho'),
            m3('mu', 'pho_mu', 'max_mu'),
            m3('rho', 'pho_rho', 'max_rho')
        ]
        arguments = {'max_iter': 200,
                     'eps':1e-7}
        def conv():
            c1 = normInfmat(variables)
        self.opt_method = Admm(mm=mm,
                               convergence=conv,
                               arguments=arguments)


    def run(self):


    def load(self,
             x,
             y=None):
        self.load_data(x)
        if y:
            self.load_label(y)

    def load_data(self,
                  x):
        self.X = x

    def load_label(self,
                   y):
        self.Y = y





