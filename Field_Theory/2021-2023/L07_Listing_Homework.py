'''
Построение графика синуса, прореживание данных (выборка каждой третьей точки),
линейная интерполяция оставшихся данных, построение двух графиков
'''
import matplotlib.pyplot as plt
import numpy as np

# https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
# 1 расчет
n = 100
m = 7
x = np.linspace(-3.5, 23.5, n); print(x)
y = np.sin(x); print(y)

plt.plot(x, y);

# 2 прореживание данных
x1 = x[0:n:m]
y1 = y[0:n:m]
print(y1)
plt.plot(x1, y1, 'ro');

#  3 линейная интерполяция
yint = np.interp(x, x1, y1)
plt.plot(x, yint, 'k');

plt.show()