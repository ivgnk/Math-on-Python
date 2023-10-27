# Решить уравнение X**X = 1 + ln(x)
# Solve the equation X**X = 1 + ln(x)
# https://www.youtube.com/watch?v=fjtq74SF-zQ

# Numeric decision

import math
import numpy as np
from FView import *

from scipy import optimize

def f0(x):
    return x**x - (1+np.log(x))

def f1(x):
    return x**x - (1+np.log(x))

def f2(x):
    return x**x - (1+math.log(x))

def f3(x):
    return x**3 - 2*x - 5

def eq2_numeric(brcktsi=None, f=f2):
    # ---- numeric decision
    if brcktsi is None:
        brcktsi = [0.0001, 3]
    # brckts = [num/10_000 for num in range(brcktsi[0],brcktsi[1]+1)]
    x = calc_arr(brcktsi, 200, 10_000)
    f = f3
    view_f1D(x,f)
    print('Gran')
    print(f'{brcktsi[0]:7}  {f(brcktsi[0])}')
    print(f'{brcktsi[1]:7}  {f(brcktsi[1])}')
    solution = optimize.root_scalar(f, bracket=brcktsi, method="bisect")
    # Решение нелинейных уравнений / Solving nonlinear equations
    # https://fadeevlecturer.github.io/python_lectures/notebooks/scipy/nonlinear_equations.html
    print(f'\n {solution=}')
    print(f'\n {solution.root=}')
    print(f'\n {f(solution.root)=}')

if __name__ == "__main__":
    eq2_numeric()



