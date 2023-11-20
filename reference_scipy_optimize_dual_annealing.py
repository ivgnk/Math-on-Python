'''
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import dual_annealing


def func(x):
    f = x * x - 10 * np.cos(2 * np.pi * x) + 10
    return f

xdata = np.linspace(-4, 4, 250)

lw = [-5.12]
up = [5.12]
ret = dual_annealing(func, bounds=list(zip(lw, up)))
print(ret)
print()
print(f'{ret.x=}')
print(f'{ret.fun}')
print(f'func={func(ret.x)}')
print(f'func={func(0)}')

ydata = func(xdata)
plt.plot(xdata, ydata, 'b-', label='data')
# plt.plot(ret.x, ret.fun,'ro', label='solution')
plt.title('scipy.optimize.dual_annealing')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
