'''
on the base https://www.indusmic.com/post/shubert-function
indusmic_shubert-function.png
usually its evaluated on the square 𝑥𝑖 ∈ [−10,10] for all i= 1,2
It has 18 global minima 𝑓(𝑥 ∗ ) ≈ −186.7309

Shubert function is continuous function.
The function is differentiable.
The function is non-separable.
The function is defined on n – dimensional space.

The Shubert function has several local minima and many global minima.
https://www.sfu.ca/~ssurjano/shubert.html
'''

import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import *

def f(x, y):
    sum1 = 0
    sum2 = 0
    for i in range(1, 6):
        sum1 = sum1 + (i * cos(((i + 1) * x) + i))
        sum2 = sum2 + (i * cos(((i + 1) * y) + i))
    return sum1 * sum2

X = np.linspace(-10,10, 100)
Y = np.linspace(-10,10, 100)

x,y = np.meshgrid(X,Y)
F = f(x,y)
minF = math.floor(F.min())
maxF = math.ceil(F.max())
print(f'{minF=}     {maxF=}')

n_iso = 15 # num isolines
# d_iso = np.logspace(minF, maxF, n_iso)
# print(f'{d_iso=}')
fig =plt.figure(figsize=(8,8))
# see also Eq_2D_002.py
curves1 = plt.contourf(x,y, F, n_iso)
curves3 = plt.contour(x, y, F, n_iso, colors='k')

fig.colorbar(curves1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Shubert function')
plt.show()
