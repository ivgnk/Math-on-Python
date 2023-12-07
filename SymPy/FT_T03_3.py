'''
For Field Theory lessons
Task 03_3:  вычисления и  построение графиков / calculations and charting
https://docs.sympy.org/latest/modules/plotting.html
Интегрирование / Integration
'''
import sympy as sym
from sympy.plotting import plot
import spb # sympy-plot-backends

x = sym.Symbol('x')
y = sym.Symbol('y')

print('\n--- (1) --- Интегрирование / Integration')
f1 = sym.integrate(sym.sin(x), x);
print(f1)   # -cos(x)
f2 = sym.integrate(sym.log(x), x);
print(f2)   # x*log(x) - x

print('\nинтегрирование: интеграл от специальной функции (функция Гаусса)')
f0 = sym.erf(x)
f1 = sym.exp(-x ** 2) * sym.erf(x);
print(f1) # exp(-x**2)*erf(x)
f2 = sym.integrate(f1, x);
print(f2) # sqrt(pi)*erf(x)**2/4
plot(f0,f1,f2, xlabel='', ylabel='', legend = True)

print('\nВычисление определенных интегралов')
r1= sym.integrate(x**3, (x, -1, 1)) # результат 0
print(r1)
print(sym.integrate(sym.sin(x), (x, 0, sym.pi / 2) )) # результат 1
print(f'{sym.integrate(sym.cos(x), (x, -sym.pi / 2, sym.pi / 2)) = }') # результат 2

print('определенные интегралы с бесконечными пределами интегрирования (несобственные интегралы)')
print(sym.integrate(sym.exp(-x), (x, 0, sym.oo)))  # результат 1





