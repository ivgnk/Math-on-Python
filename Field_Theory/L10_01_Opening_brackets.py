'''
Opening brackets / Раскрытие скобок
'''
import matplotlib.pyplot as plt
import numpy as np
import math

import sympy as sym

print('Example')
x = sym.Symbol('x')
y = sym.Symbol('y')
z = sym.Symbol('z')
v = sym.Symbol('v')
print( sym.expand((x + y) ** 3) )
# При помощи ключевого слова можно добавить поддержку работы с тригонометрическими функциями
print( sym.expand(sym.cos(x + y), trig=True))

print('\nTask')
print( sym.expand((sym.sin((x + 2)**2)), trig=True))
print('\n')


X = np.linspace(-32,32)
Y = np.linspace(-32,32)
x,y = np.meshgrid(X,Y)

def f(x, y):
    return np.power(x+y,2)

F = f(x,y)

n_iso = 15 # num isolines
fig =plt.figure(figsize=(10,8))
curves1 = plt.contourf(x,y, F, n_iso)
# curves3 = plt.contour(x, y, F, n_iso, colors='k')

fig.colorbar(curves1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sphere function')
plt.show()

