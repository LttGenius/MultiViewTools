class Module:
    compute_value = 0
    variables = {}
    arguments = []
    opt_variable = ""

    def __init__(self,
                 variable,
                 *arg):
        self.opt_variable = variable
        self.arguments = arg

    def load_variable(self,
                      variable_pool):
        self.variables = variable_pool

    def optimization(self,
                     arg=None):
        pass

    def value(self):
        return None


