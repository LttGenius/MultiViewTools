class Algorithm:
    mm = []
    variables_pool = {}
    conv_cure = []
    hyper_param = {}
    conv = None

    def upload(self,
               mm):
        for i in mm:
            i.load_variable(self.variables_pool)
            self.mm.append(i)

    def upload_conv(self,
                    conv):
        self.conv = conv
        self.conv.load_variable(self.variables_pool)

    def __rshift__(self,
                   other):
        self.upload_conv(other)

    def __lshift__(self,
                   other):
        self.upload([other])
        return self

    def __init__(self,
                 variables,
                 mm=None):
        self.variables_pool = variables
        if mm:
            self.upload(mm)

    def optimization(self):
        pass

    def __check_convergence(self):
        pass

