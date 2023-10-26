# Что больше pi**2 or 2**pi
# https://www.youtube.com/watch?v=g6xZ8qW05mA

import math
import matplotlib.pyplot as plt
import numpy as np


d1:float = math.pi**2
d2:float = 2**math.pi

print(f' {math.pi=}')
print(f' math.pi**2 = {d1}   2**math.pi = {d2} ')

# ---- graphs decision
x:np.ndarray = np.linspace(-6, 6,200)
y1:np.ndarray = x**2
y2:np.ndarray = 2**x

plt.plot(x, y1, label = 'x**2')
plt.plot(x, y2, label = '2**x')

x_l = [math.pi, math.pi]
y_l = [0,60]
plt.plot(x_l, y_l,label = 'math.pi')


plt.grid()
plt.legend()
plt.show() # показываем график
