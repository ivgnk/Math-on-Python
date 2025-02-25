"""
https://studfile.net/preview/4447247/
стр.14
Применим понятие решение к дифференциальному уравнению
v'=g,
описывающему свободное падение тела в поле тяготения Земли.
"""
from  sympy  import *
t = symbols('t')
g = symbols('g')
v = Function('v')
# Объявление символьной функции
# f = v' - g
f = v(t).diff(t)-g
print('Исходное уравнение =\n', f)  #
f1 = dsolve(f, v(t))
print('Решение уравнения =\n', f1)



