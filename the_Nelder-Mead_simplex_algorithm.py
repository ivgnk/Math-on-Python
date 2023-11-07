# Based on https://docs.scipy.org/doc/scipy/tutorial/optimize.html#nelder-mead-simplex-algorithm-method-nelder-mead
# also https://en.wikipedia.org/wiki/Rosenbrock_function

# https://habr.com/ru/articles/439288/

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen2(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x0  = np.array([100, 100]); res = 1e10; it = 0

while (res >1e5):
    print()
    result = minimize(rosen, x0, method='nelder-mead',
                   options={'xatol': 1e-8, 'disp': True})
    print(result)
    res = result.fun
    x0 = result.x
    print(f'{it=}')
    input()
    it = it+1


# edge = 2
# num_ = 6
# x1 = np.linspace(-2, 2, num_)
# x2 = np.linspace(-2, 2, num_)
# # https://stackoverflow.com/questions/28578302/how-to-multiply-two-vector-and-get-a-matrix
# matr = np.outer(x1,x2)
# for i in range(num_):
#     for j in range(num_):
#         # matr[i,j] = x1[i]*x2[j]
#         x = np.array([x1[i], x2[j]])
#         matr[i, j] = x1[i]*x2[j] # rosen(x)
#         if (x1[i] > 0) :
#             matr[i, j] = matr[i, j]*2
#
# print(matr)
# curves2 = plt.contourf(x1, x2, matr)
# curves3 = plt.contour(x1, x2, matr, colors = 'k')
# plt.clabel(curves3, fontsize=10, colors = 'k')
# plt.show()