'''
on the base
https://ru.wikipedia.org/wiki/Тестовые_функции_для_оптимизации#Тестовые_функции_для_одной_цели_оптимизации
'''

import matplotlib.pyplot as plt
import numpy as np
import math

def f(x, y):
    return x ** 2 + y ** 2

X = np.linspace(-32,32)
Y = np.linspace(-32,32)
x,y = np.meshgrid(X,Y)
F = f(x,y)
minF = F.min()
maxF = F.max()
print(minF,maxF)

n_iso = 15 # num isolines
fig =plt.figure(figsize=(10,8))
curves1 = plt.contourf(x,y, F, n_iso)
curves3 = plt.contour(x, y, F, n_iso, colors='k')

fig.colorbar(curves1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sphere function')
plt.show()


