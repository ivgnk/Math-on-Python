'''
2017_Язык Python 3 для научных исследований_Долгих.pdf
page 71
'''

import sympy as sym
x = sym.Symbol('x')
f = sym.symbols('f', cls=sym.Function)

f1 = sym.dsolve( f(x).diff(x, x) + f(x).diff(x)-2*(x+1), f(x) )
print(f1) # решение уравнения
# Eq(f(x), C1 + C2*exp(-x) + x**2) - решение уравнения