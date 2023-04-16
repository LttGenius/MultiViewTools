from base.support import Module


class Convergence(Module):
    def __init__(self, 
                 *arg):
        super().__init__("__CONVERGENCE__", *arg)

    def compute(self):
        return self.optimization()
    
