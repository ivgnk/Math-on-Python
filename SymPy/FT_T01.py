'''
For Field Theory lessons
Task 01: Запись переменных, их вычисление
'''

import sympy as sym
import math

# В библиотеке SymPy есть три встроенных численных типа данных: Real, Rational и Integer.
# Тип Rational представляет рациональное число как пару чисел: числитель и знаменатель рациональной дроби.
# Сравним обычный тип float в Python и Rational из sympy

# здесь различий как бы нет
a_s = sym.Rational(1, 2); a_p: float = 0.5
print(a_s,'  ',a_p)    # 1/2   0.5

# здесь различия видны
a_s = sym.Rational(1, 3); a_p: float = 1/3
print(a_s,'  ',a_p)   # 1/3   0.3333333333333333

# Сравнение pi из math и sympy
print(math.pi,'  ',sym.pi)  # 3.141592653589793    pi

# Сравнение pi из math и вычисленный sympy
print(math.pi,'  ',sym.pi.evalf())

# Вычисление констант с разным  числом значащих цифр
print('\nВычисление констант с разным  числом значащих цифр')
n = 10 # максим. число значащих цифр для  вычисления
for i in range(n):
    print(i,'  ',sym.pi.evalf(i))

# Обозначение бесконечности sym.oo (две латинских o подряд)
print('\nСимвольная бесконечнность')
print(sym.oo > 99999)  # True
print(sym.oo + 1)      # oo

# Задание и использование символьных переменных
print('\nЗадание и использование символьных переменных')
x = sym.Symbol('x')
y = sym.Symbol('y')
print( x + y + x - y )  # 2*x
print( (x + y) ** 2 )   # (x + y)**2