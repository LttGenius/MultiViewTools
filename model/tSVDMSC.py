from base.tensor import TNN
from base import normInfmat
from base import Admm
from base import Spare21Norm
from base import Convergence
from base import Module
from base import Connect
from model import base_model
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


class conv(Convergence):
    def optimization(self, 
                     arg=None):
        var = self.variables[self.arguments[0]] - np.matmul(self.variables[self.arguments[0]], 
                                                            self.variables[self.arguments[1]]) - \
        self.variables[self.arguments[2]]
        c1 = normInfmat(var)
        var = self.variables[self.arguments[3]] - self.variables[self.arguments[1]]
        c2 = normInfmat(var)
        if c1 < arg and c2 < arg:
            t1 = True
        else:
            t1 = False
        return c1, c2, t1
        


class tSVDMSC(base_model):
    def load_mm(self):
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
        # mm = [
        #     c1(),
        #     m1('Z', 'N', 'mu', 'rho'),
        #     c2(),
        #     Spare21Norm('E', 'lambda', 'mu'),
        #     c3(),
        #     m2('Y', 'mu'),
        #     c4(),
        #     TNN('G', 'rho'),
        #     c5(),
        #     m2('W', 'rho'),
        #     m3('mu', 'pho_mu', 'max_mu'),
        #     m3('rho', 'pho_rho', 'max_rho')
        # ]
        arguments = {'max_iter': 200,
                     'eps':1e-7}
        self.opt_method = Admm( arguments=arguments )
        self.opt_method << c1() << m1('Z', 'N', 'mu', 'rho') << c2() << Spare21Norm('E', 'lambda', 'mu') \
                   << c3() << m2('Y', 'mu') << c4() << TNN('G', 'rho') << c5() << m2('W', 'rho') \
                   << m3('mu', 'pho_mu', 'max_mu') << m3('rho', 'pho_rho', 'max_rho') \
                   >> conv('X', 'Z', 'E', 'G')
        

def run_tSVDMSC(x: np.ndarray,
                y: np.ndarray = None,
                arguments: dict = None):
    if not arguments:
        arguments['lambda'] = 0.5
    model_tsvdmsc = tSVDMSC(x=x,
                            y=y,
                            argument=arguments)
    var = model_tsvdmsc.run()
    z = var['Z']
    v = z.shape[0]
    s = 0
    for i in range(v):
        tmp = z[i, :, :]
        s = s + (np.abs(tmp) + np.abs(tmp).T) / 2
    
    



