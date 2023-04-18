from base.tensor import TNN
from base import normInfmat
from base import Admm
from base import Spare21Norm
from base import Module
from base import Connect
from base import Convergence
import numpy as np



class base_model:
    X = 0
    Y = 0
    mm = []
    arg = {}
    opt_method = None
    opt_method_name = None
    __opt_methods_dict = {"ADMM": Admm}
    def __init__(self,
                 x: np.ndarray = None,
                 y: np.ndarray = None,
                 argument: dict = None,
                 opt_method = "ADMM") -> None:
        self.load(x, y)
        if argument:
            self.load_arg(argument)
        self.opt_method_name = opt_method
        self.__load_opt_method()
    
    def load_arg(self,
                 argument: dict):
        self.arg = argument

    def load(self,
             x: np.ndarray = None,
             y: np.ndarray = None):
        if x:
            self.load_data(x)
        if y:
            self.load_label(y)

    def load_data(self,
                  x: np.ndarray):
        self.X = x
    
    def load_label(self,
                  y: np.ndarray):
        self.Y = y
    
    def __load_opt_method(self):
        self.opt_method = self.__opt_methods_dict[self.opt_method_name]
    
    def load_mm(self):
        pass

    def run(self):
        self.opt_method.optimization()
        return self.get_variable()
    
    def get_variable(self):
        return self.opt_method.get_variables()

    

        
        
        