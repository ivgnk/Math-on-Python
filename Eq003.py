# https://www.youtube.com/watch?v=cX5_eqfj2BE
# X**X +(1/X)**(1/X) = 4 +(1/2)^(1/2)

import math
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

def f1float(x:float)->float:
    return  x ** x + (1 / x) ** (1 / x) - (4 + 0.5 ** 0.5)

def f1(x:np.ndarray)->np.ndarray:
    return  x ** x + (1 / x) ** (1 / x) - (4 + 0.5 ** 0.5)

def f2(x:np.ndarray)->np.ndarray:
    xx = deepcopy(x); llen = x.size
    print(llen)
    for i in range (llen):
        xx[i] = x[i]**x[i] + (1/x[i])**(1/x[i]) - (4 + 0.5 ** 0.5)
    return xx

# ---- graphs decision
x:np.ndarray = np.linspace(0.3, 3.0,200)
y1:np.ndarray = f1(x)
y2:np.ndarray = f2(x)
plt.plot(x, y2, label = 'for')
plt.plot(x, y1, label = 'vec', linestyle='dashed')
plt.grid()
print(f'{f1float(0.5)=}')
print(f'{f1float(2)=}')

x_l1 = [0.5, 0.5]; x_l2 = [2, 2]
y_l = [-5,60]
plt.plot(x_l1, y_l,label = 'root 1', color = 'green')
plt.plot(x_l2, y_l,label = 'root 2', color = 'green', linestyle='dashed')
plt.legend()
plt.show()

# https://fadeevlecturer.github.io/python_lectures/notebooks/scipy/nonlinear_equations.html
# solution = optimize.root_scalar(f1, bracket=[0.01, 10], method="bisect")
# ValueError: f(a) and f(b) must have different signs
# print(solution)