"""
https://habr.com/ru/articles/423731/
В функциях symbols() и var() можно объявлять символьные переменные с индексом
"""
#----------------- 1
# from sympy import *
# x=symbols('x:9'); print(x)  # диапазон индексов от 0 до 8
# x=symbols('x5:10'); print(x)  # диапазон индексов от 5 до 9
# x=var('x:9'); print(x)   # диапазон индексов от 0 до 8
# x=var('x5:10'); print(x)   # диапазон индексов от 5 до 9

#----------------- 2
# from sympy import *
# x = symbols('x', integer=True) #назначаем целый тип
# print(sqrt(x**2))
# x = symbols('x', positive = True, integer=True)
# print(sqrt(x**2))
# x = symbols('x')
# print(sqrt(x**2)) # это x, если x≥0

#----------------- 3
# from sympy import *
# x, y = symbols('x y')
# expr = x**2 + sin(y) + S(10)/2;
# print(expr, type(expr)) # выполнено деление 10/2
# print(type(10))  # <class 'int'>
# s10=S(10)
# print(s10, type(s10)) # символьная константа 10 и ее тип

#----------------- 4
from sympy import *
z=1/7    # переменная Python
print(f'Python {z=}') # вычисляет переменную z с процессорной точностью 0.14285714285714285
z1=S(1)/7 # переменная SymPy
print(f' SymPy {z1=}') # z1
z2=z1.n(30) # вычисляет переменную z2 с точностью до 30 значащих цифр
print(f' SymPy округл SymPy {z2=}') # 0.142857142857142857142857142857
z3=round(z1,30) # встроенная функция Python, округляем до 30 цифр
print(f'Python округл SymPy {z3=}') # 0.142857142857142857142857142857
z4=round(z,30) # встроенная функция Python, округляем до 30 цифр
print(f'Python округл Python {z4=}')

# print(round(1/7,40))