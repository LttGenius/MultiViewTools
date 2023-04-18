class aa:
    def __init__(self):
        self.x = 0
    def cc(self):
        self.x += 1


def fun(a, *arg):
    aa = arg[0]
    aa.cc()

asdasa = aa()
fun(1, asdasa)
fun(1, asdasa)
fun(1, asdasa)
print(asdasa.x)