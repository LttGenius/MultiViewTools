from base.support import tensor2matrix, matrix2tensor
from base.support import Module
from base.tensor import tensorFFT
from base.tensor import tensorIFFT
import numpy as np


def shiftdim(ten,
             mode):
    if mode == 1:
        return np.rollaxis(np.rollaxis(ten, 0, 2), 2, 1)
    if mode == 2 or mode == -2:
        return np.rollaxis(ten, 1, 0)
    if mode == 0:
        return ten
    if mode == -1:
        return np.rollaxis(ten, 2, 0)


def soft(mat,
         v):
    tmp = mat - v
    tmp[tmp < 0] = 0
    return tmp


class TensorNuclearNorm(Module):
    rotate_way = {0: lambda x: shiftdim(x, 0),
                  1: lambda x: shiftdim(x, 1),
                  2: lambda x: shiftdim(x, 2)}
    def optimization(self, 
                     arg=None):
        sx = self.variables[self.opt_variable].shape
        rho = 1/self.variables[self.arguments[0]]
        if self.arguments[1] and self.arguments[1] in self.variables:
            rotate = self.variables[self.arguments[1]]
        else:
            rotate = 2
        if self.arguments[2] and self.arguments[2] in self.variables:
            is_weight = self.variables[self.arguments[2]]
            C = np.sqrt(sx[0]*sx[2])
        else:
            is_weight = False
            C = 0
        n3 = sx[0]
        Y = self.rotate_way[rotate](self.variables[self.opt_variable])
        Yhat = tensorFFT(Y)
        eps = 1e-6
        objV = 0
        if not n3%2:
            end = n3//2 + 1
            for i in range(end):
                uhat, shat, vhat = np.linalg.svd(Yhat[i, :, :])
                if is_weight:
                    w = C/(shat + eps)
                    tau = rho * w
                    shat = soft(shat, tau)
                else:
                    tau = rho
                    shat = soft(shat, tau)
                objV = objV + np.sum(shat)

                Yhat[i, :, :] =  np.matmul(np.matmul(uhat, shat), vhat)
                if i > 0:
                    Yhat[n3-i+1, :, :] = np.matmul(np.matmul(np.conj(uhat), shat), vhat)
                    objV = objV + np.sum(shat)
            uhat, shat, vhat = np.linalg.svd(Yhat[end, :, :])
            if is_weight:
                w = C / (shat + eps)
                tau = rho * w
                shat = soft(shat, tau)
            else:
                tau = rho
                shat = soft(shat, tau)
            objV = objV + np.sum(shat)
            Yhat[end, :, :] = np.matmul(np.matmul(uhat, shat), vhat)
        else:
            end = n3 // 2 + 1
            for i in range(end):
                uhat, shat, vhat = np.linalg.svd(Yhat[i, :, :])
                if is_weight:
                    w = C / (shat + eps)
                    tau = rho * w
                    shat = soft(shat, tau)
                else:
                    tau = rho
                    shat = soft(shat, tau)
                objV = objV + np.sum(shat)

                Yhat[i, :, :] = np.matmul(np.matmul(uhat, shat), vhat)
                if i > 0:
                    Yhat[n3 - i + 1, :, :] = np.matmul(np.matmul(np.conj(uhat), shat), vhat)
                    objV = objV + np.sum(shat)
        Y = tensorIFFT(Yhat)
        Y = shiftdim(Y, -rotate)
        self.compute_value = objV
        self.variables[self.opt_variable] = Y

    def value(self):
        return self.compute_value


class WeightTensorNuclearNorm(Module):
    rotate_way = {0: lambda x: shiftdim(x, 0),
                  1: lambda x: shiftdim(x, 1),
                  2: lambda x: shiftdim(x, 2)}

    def optimization(self,
                     arg=None):
        sx = self.variables[self.opt_variable].shape
        rho = self.variables[self.arguments[0]]
        n3 = sx[0]
        rotate = 1
        Y = self.rotate_way[rotate](self.variables[self.opt_variable])
        Yhat = tensorFFT(Y)
        eps = 1e-6
        objV = 0
        if not n3 % 2:
            end = n3 // 2 + 1
            for i in range(end):
                uhat, shat, vhat = np.linalg.svd(Yhat[i, :, :])
                tau = rho
                shat = soft(shat, tau)
                objV = objV + np.sum(shat)

                Yhat[i, :, :] = np.matmul(np.matmul(uhat, shat), vhat)
                if i > 0:
                    Yhat[n3 - i + 1, :, :] = np.matmul(np.matmul(np.conj(uhat), shat), vhat)
                    objV = objV + np.sum(shat)
            uhat, shat, vhat = np.linalg.svd(Yhat[end, :, :])

            tau = rho
            shat = soft(shat, tau)
            objV = objV + np.sum(shat)
            Yhat[end, :, :] = np.matmul(np.matmul(uhat, shat), vhat)
        else:
            end = n3 // 2 + 1
            for i in range(end):
                uhat, shat, vhat = np.linalg.svd(Yhat[i, :, :])

                tau = rho
                shat = soft(shat, tau)
                objV = objV + np.sum(shat)

                Yhat[i, :, :] = np.matmul(np.matmul(uhat, shat), vhat)
                if i > 0:
                    Yhat[n3 - i + 1, :, :] = np.matmul(np.matmul(np.conj(uhat), shat), vhat)
                    objV = objV + np.sum(shat)
        Y = tensorIFFT(Yhat)
        Y = shiftdim(Y, -rotate)
        self.compute_value = objV
        self.variables[self.opt_variable] = Y

    def value(self):
        return self.compute_value

def gemman(x,
           theta):
    return (1+theta) * theta / (theta + x) ** 2
def general(x,
            theta):
    return theta


class NonconvexTensorNuclearNorm(Module):
    non_fun = {'gemman': gemman,
               'general': general}
    def optimization(self,
                     arg=None):
        sx = self.variables[self.opt_variable].shape
        nonconvex_fun = self.variables[self.arguments[0]]
        theta = self.variables[self.arguments[1]]
        nonFun = self.non_fun[nonconvex_fun]
        n3 = sx[0]
        rotate = 1
        Y = shiftdim(self.variables[self.opt_variable], 1)
        Yhat = tensorFFT(Y)
        objV = 0
        if not n3 % 2:
            end = n3 // 2 + 1
            for i in range(end):
                uhat, shat, vhat = np.linalg.svd(Yhat[i, :, :])
                tau = nonFun(shat, theta)
                shat = soft(shat, tau)
                objV = objV + np.sum(shat)

                Yhat[i, :, :] = np.matmul(np.matmul(uhat, shat), vhat)
                if i > 0:
                    Yhat[n3 - i + 1, :, :] = np.matmul(np.matmul(np.conj(uhat), shat), vhat)
                    objV = objV + np.sum(shat)
            uhat, shat, vhat = np.linalg.svd(Yhat[end, :, :])

            tau = nonFun(shat, theta)
            shat = soft(shat, tau)
            objV = objV + np.sum(shat)
            Yhat[end, :, :] = np.matmul(np.matmul(uhat, shat), vhat)
        else:
            end = n3 // 2 + 1
            for i in range(end):
                uhat, shat, vhat = np.linalg.svd(Yhat[i, :, :])

                tau = nonFun(shat, theta)
                shat = soft(shat, tau)
                objV = objV + np.sum(shat)

                Yhat[i, :, :] = np.matmul(np.matmul(uhat, shat), vhat)
                if i > 0:
                    Yhat[n3 - i + 1, :, :] = np.matmul(np.matmul(np.conj(uhat), shat), vhat)
                    objV = objV + np.sum(shat)
        Y = tensorIFFT(Yhat)
        Y = shiftdim(Y, -rotate)
        self.compute_value = objV
        self.variables[self.opt_variable] = Y

    def value(self):
        return self.compute_value
        
        