# https://www.youtube.com/watch?v=eEEBW6CN2E8
# X**(X**X) = (1/2)^Sqrt(2)

import math
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

the_const: float = (1/2)**math.sqrt(2)

def f1float(x:float)->float:
    return  x ** (x**x) - the_const

def f1(x:np.ndarray)->np.ndarray:
    return  x ** (x**x) - the_const

def f2(x:np.ndarray)->np.ndarray:
    xx = deepcopy(x); llen = x.size
    print(llen)
    for i in range (llen):
        xx[i] = x[i]**(x[i]**x[i]) - the_const
    return xx

# ---- graphs decision
br = [0.1, 3.0]
x:np.ndarray = np.linspace(0.1, 2.0,200)
y1:np.ndarray = f1(x)
y2:np.ndarray = f2(x)
plt.plot(x, y2, label = 'for')
plt.plot(x, y1, label = 'vec', linestyle='dashed')
plt.grid()
# print(f'{f1float(0.5)=}')
# print(f'{f1float(2)=}')

# x_l1 = [0.5, 0.5]; x_l2 = [2, 2]# https://fadeevlecturer.github.io/python_lectures/notebooks/scipy/nonlinear_equations.html
solution = optimize.root_scalar(f1, bracket=br, method="bisect")
print(solution)
x_l1 = [0.25, 0.25]
y_l = [-5,60]
plt.plot(x_l1, y_l,label = 'root 1', color = 'green')
plt.legend()
plt.show()

