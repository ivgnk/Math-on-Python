# Решить уравнение X**X = 1 + ln(x)
# Solve the equation X**X = 1 + ln(x)
# https://www.youtube.com/watch?v=fjtq74SF-zQ

from Eq002_F1 import *

brcktsi:list = [1, 30000]
# eq2_fraph(brcktsi)

# ---- Scipy decision
# Поиск корня скалярной функции одного аргумента / Finding the root of a scalar function of one argument
# https://fadeevlecturer.github.io/python_lectures/notebooks/scipy/nonlinear_equations.html

import scipy

# print(f'{brckts[0]:7}  {f1(brckts[0])}')
# print(f'{brckts[1]:7}  {f1(brckts[1])}')


import sympy
x = sympy.symbols('x')
solution = sympy.solve(x**x - 1 -  ,x ) # - sympy.log(x)

print(f'\n {solution=}')
