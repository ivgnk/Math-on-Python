"""
Символьные вычисления средствами Python. Часть1. Основы
Явное объявление символьных переменных
https://habr.com/ru/articles/423731/
"""
from sympy import *
print('1. Способы задания')
print(f'{type(symbols)=}  {type(Symbol)=}  {type(var)=} \n')
# созданы четыре символьные переменные,
# предыдущие значения переменных затираются
print('2. symbols и Symbol - способы задания')
x,y,a,b = symbols('x y a b')
print(f'{x=} {y=} {a=} {b=}')
x,y,a,b = symbols('x, y, a, b')
print(f'{x=} {y=} {a=} {b=}')
x,y,a,b = symbols('x, y  a  b')
c=Symbol('c')
print(f'{type(a)=}, {type(c)=} \n')
print('3. var - способы задания')
var('u')
var('v w')
print(f'{a=}, {type(a)=}')
print(f'{v=}, {type(v)=} \n')
print('4. var - способы задания "непрямые"')
a=x
print(f'{a=}, {type(a)=}')
a=x+y
print(f'{a=}, {type(a)=}')
a=2*x
print(f'{a=}, {type(a)=}')

