from SRacos import SRacos


class Sphere:
    def __init__(self):
        pass

    def eval(self, x):
        s = 0.0
        for t in x:
            s += t * t
        return s ** 0.5


# dimension = [[-3, 3, True, True, None],
#              [-3, 3, True, True, None],
#              [-3, 3, True, True, None]]
# dimension = [[-50, 70, False, True, [3, -2, 1, 7]],
#              [-30, 30, False, True, [-1, 2, -3, 3]],
#              [-60, 50, False, True, [4, 5, -2, 1]]]
dimension = [[-50, 70, False, False, [3, -2, 1, 7]],
             [-30, 30, False, False, [-1, 2, -3, 3]],
             [-60, 50, False, False, [4, 5, -2, 1]]]
sracos = SRacos()
x, y = sracos.opt(Sphere(), dimension, 10000, 10, 20, 0.5, 1)
print(x, y)
