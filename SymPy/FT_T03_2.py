'''
For Field Theory lessons
Task 03_2:  вычисления и  построение графиков / calculations and charting
https://docs.sympy.org/latest/modules/plotting.html
Разложение в ряд / Decomposition in a row
'''
import sympy as sym
from sympy.plotting import plot
import spb # sympy-plot-backends

x = sym.Symbol('x')
y = sym.Symbol('y')

print('\n--- (1) --- Разложение в ряд')
# Для разложения выражения в ряд Тейлора используется функция: series(expr, var)
f = sym.cos(x)
n = 9
rng = range(n)
lst = list(rng)
for i in rng:
    lst[i-1] = sym.series(sym.cos(x), x, n = i)
    print(f'{i}   {lst[i-1]}' )
# 0   O(1)
# 1   1 + O(x)
# 2   1 + O(x**2)
# 3   1 - x**2/2 + O(x**3)
# 4   1 - x**2/2 + O(x**4)
# 5   1 - x**2/2 + x**4/24 + O(x**5)
# 6   1 - x**2/2 + x**4/24 + O(x**6)
# 7   1 - x**2/2 + x**4/24 - x**6/720 + O(x**7)
# 8   1 - x**2/2 + x**4/24 - x**6/720 + O(x**8)