from base.support import tensor2matrix, matrix2tensor
from base.support import Module
import numpy as np


class Spare21Norm(Module):
    def optimization(self,
                     arg=None):
        tmp = tensor2matrix(arg, 2)
        nw = np.sum(tmp**2, axis=1)**0.5
        tt = self.variables[self.arguments[0]] / self.variables[self.arguments[1]]
        nw = (nw - self.variables[self.arguments[0]]) / nw
        nw[nw < 0] = 0
        tmp = tmp * nw
        self.variables[self.opt_variable] = matrix2tensor(tmp.T, self.variables[self.opt_variable].size(), 2)

    def value(self):
        tmp = tensor2matrix(self.variables[self.opt_variable], 2)
        self.compute_value = np.sum(np.sum(tmp**2, axis=0)**0.5)
        return self.compute_value


class Spare1Norm(Module):
    def optimization(self,
                     arg=None):
        t = np.abs(arg) - self.variables[self.arguments[0]]
        t[t < 0] = 0
        self.variables[self.opt_variable] = np.sign(arg) * t

    def value(self):
        tmp = tensor2matrix(self.variables[self.opt_variable], 2)
        self.compute_value = np.max(np.sum(np.abs(tmp), axis=0))
        return self.compute_value


class SpareCauthyNorm(Module):
    def optimization(self,
                     arg=None):
        tmp_lambda = self.variables[self.arguments[0]]
        noise_lambda = self.variables[self.arguments[0]]*self.variables[self.arguments[1]]
        t = self.variables[self.arguments[2]] / (self.variables[self.arguments[2]] - noise_lambda) * arg
        noise_lambda = self.variables[self.arguments[2]]/noise_lambda - 1
        tmp = tensor2matrix(t, 2)
        nw = np.sum(tmp**2)**0.5
        nw = (nw - noise_lambda) / nw
        nw[nw < 0] = 0
        tmp = tmp * nw
        tmpnw = np.sum(tmp**2)**0.5
        self.compute_value = tmp_lambda * np.sum(np.log(1 + self.variables[self.arguments[1]]*tmpnw))
        self.variables[self.opt_variable] = matrix2tensor(tmp.T, self.variables[self.opt_variable].size(), 2)

    def value(self):
        return self.compute_value
