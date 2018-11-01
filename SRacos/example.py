from SRacos import SRacos


def func(x):
    s = 0.0
    for t in x:
        s += t * t
    return s ** 0.5


# dimension = [[-3, 3, SRacos.CONTINUOUS, None],
#              [-3, 3, SRacos.CONTINUOUS, None],
#              [-3, 3, SRacos.CONTINUOUS, None]]
# dimension = [[-50, 70, SRacos.DISCRETE, [3, -2, 1, 7]],
#              [-30, 30, SRacos.DISCRETE, [-1, 2, -3, 3]],
#              [-60, 50, SRacos.DISCRETE, [4, 5, -2, 1]]]
dimension = [[-50, 70, SRacos.CONTINUOUS, [3, -2, 1, 7]],
             [-30, 30, SRacos.CATEGORICAL, [-1, 2, -3, 3]],
             [-60, 50, SRacos.CATEGORICAL, [4, 5, -2, 1]]]
sracos = SRacos.Optimizer()
x, y = sracos.opt(func, dimension, 1000, 10, 20, 0.5, 1)
print(x, y)
