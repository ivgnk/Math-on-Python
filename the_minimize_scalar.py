#### SciPy API
# Optimization and root finding (scipy.optimize)
# https://docs.scipy.org/doc/scipy/reference/optimize.html

# Scalar functions optimization
# minimize_scalar(fun[, bracket, bounds, ...])
# Minimization of scalar function of one variable.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html#scipy.optimize.minimize_scalar

#### USER GUIDE (tutorial)
# https://docs.scipy.org/doc/scipy/tutorial/optimize.html
# Univariate function minimizers (minimize_scalar)
# Unconstrained minimization (method='brent')
# minimize_scalar(method=’golden’) - golden section method

# https://docs.scipy.org/doc/scipy/tutorial/optimize.html#bounded-minimization-method-bounded
# minimize_scalar(method='bounded') - The interval constraint allows the minimization
# to occur only between two fixed endpoints, specified using the mandatory bounds parameter.

# Вывод по optimize.minimize_scalar(f, method='brent')
# Unconstrained minimization (method='brent')
# Uses inverse parabolic interpolation when possible to speed up convergence of golden section method
# 1) Поиск очень плохой, действительно для унимодальной функции
# 2) Скатывание в ближайший локальный минимум при поиске для неунимодальной функции
# 3) Задание bracket не является реальным ограничением, поиск выходит за эти границы
# 4) Нет задания области определения функции, т.к. bracket не работают
# 5) Без bracket ищет минимум вблизи нуля

# Вывод по optimize.minimize_scalar(f, method='golden')
# 1) Все тоже самое, что и 'brent', который является разновидностью 'golden'
# 2) При задании bracket=(-2, -1) нет ошибки untimeWarning: overflow encountered in scalar multiply return (x - 2) * (x + 1)**2

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.optimize as optimize

from FView import *

num_=140; edge = 5
x = np.linspace(-edge, edge, num_)

def funct_x3_1_(x):
    return x+23*x**2 -2*x**3

def funct_x4_1_(x):
    return (x - 2) * (x + 1)**2

f = funct_x4_1_
fig, ax = plt.subplots(figsize=(10, 6))

xlim = -3, 3
ax.set_xlim(xlim)
ylim = f(-3), f(3)
ax.set_ylim(ylim)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))

met = 'bounded'
b1= (-2, -1)
res_bounded = optimize.minimize_scalar(f, method=met, bounds=b1)
print(res_bounded)
xres1 = res_bounded.x
yres1 = res_bounded.fun
print('------------------')
b2 = (0, 2)
res_bounded = optimize.minimize_scalar(f, method=met, bounds=b2)
print(res_bounded)
xres2 = res_bounded.x
yres2 = res_bounded.fun

plt.title('optimize.minimize_scalar(f, method="bounded", bounds=gr1, gr2)')
y = f(x); plt.plot(x,y,label = 'function'); plt.grid()
plt.plot(xres1, yres1,'ro',label = 'min 1')
plt.plot(xres2, yres2,'bo',label = 'min 2')
make_lines(b1,ylim)
make_lines(b2,ylim)
plt.legend()
plt.show()










