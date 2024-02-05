import sympy as sym
import math
print(math.pi,'  ',sym.pi) # 3.141592653589793    pi
# Сравнение pi из math и вычисленный sympy
print(math.pi,'  ',sym.pi.evalf())

print('\n')
# Вычисление констант с разным числом значащих цифр
n = 10 # Максимальное число значащих цифр для вычисления
for i in range(n):
    print(i,'  ',sym.pi.evalf(i))
print('\n')

print(sym.oo > 99999)
print( sym.oo + 1)
