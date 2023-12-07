'''
For Field Theory lessons
Task 02: Преобразования и вычисление
'''

import sympy as sym
import math

print('--- (1) --- Раскрытие скобок')
x = sym.Symbol('x')
y = sym.Symbol('y')
print( sym.expand((x + y) ** 3) )  # x**3 + 3*x**2*y + 3*x*y**2 + y**3
print( sym.expand(sym.cos(x + y), trig=True)) # -sin(x)*sin(y) + cos(x)*cos(y)

print('\n--- (2) --- Упрощение выражений')
print(f'{sym.simplify((x + x * y) / x) = }' )   #  y+1

print('\n--- (3) --- Вычисления пределов')
# limit(function, variable, point)
# если хотите вычислить предел
# функции f(x), где x -> 0, то надо написать limit(f(x), x, 0).
print(f'{sym.limit(sym.sin(x)/x, x, 0) =}')  #  1
print()
print(f'{sym.limit(x, x, sym.oo) = }' )  # oo
print(f'{sym.limit(1 / x, x, sym.oo) = }') # 0
print(f'{sym.limit(x ** x, x, 0) = }') # 1

print('\n--- (4) --- Дифференцирование')
print(f'{sym.diff(sym.sin(x), x) = }' )   # cos(x)
print(f'{sym.diff(sym.sin(2 * x), x) = }')  # 2*cos(2*x)
print(f'{sym.diff(sym.tan(x), x) = }')    # tan(x)**2 + 1



