from base.support import Algorithm


class Admm(Algorithm):
    max_iter = 200
    eps = 1e-6
    now_iter = 0

    def load_arguments(self,
                       arguments: dict):
        if "max_iter" in arguments:
            self.max_iter = arguments["max_iter"]
        if "eps" in arguments:
            self.eps = arguments['eps']

    def __init__(self,
                 variable,
                 mm=None,
                 convergence=None,
                 arguments=None):
        super(Admm, self).__init__(variable, mm)
        if convergence:
            self.upload_conv(convergence)
        if arguments:
            self.load_arguments(arguments=arguments)

    def __check_convergence(self):
        t = self.conv.compute(self.eps)
        self.conv_cure.append(t)
        if self.now_iter > self.max_iter:
            return True
        return t[-1]

    def optimization(self):
        self.now_iter = 0
        t = None
        for m in self.mm:
            t = m.optimization(t)
        self.now_iter += 1
        while not self.__check_convergence():
            for m in self.mm:
                t = m.optimization(t)
            self.now_iter += 1
    









