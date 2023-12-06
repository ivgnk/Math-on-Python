'''
on the base https://www.indusmic.com/post/ackleyn-2function
Mathematical Definition f(x,y) =-200*exp(-0.2*sqrt(x**2+y**2))
Input Domain x ∈ [−32, 32] and y ∈ [−32, 32]
The global minimum of the function is at f(x∗ ) = −200 located at x ∗ = (0, 0)

easy variant from Ackley_function for testing optimization algoritms
see
https://en.wikipedia.org/wiki/Ackley_function
https://en.wikipedia.org/wiki/Test_functions_for_optimization
'''

import matplotlib.pyplot as plt
import numpy as np
import math

def f(x, y):
    return -200 * np.exp(-0.2 * np.sqrt(x ** 2 + y ** 2))

X = np.linspace(-32,32)
Y = np.linspace(-32,32)

x,y = np.meshgrid(X,Y)
F = f(x,y)
minF = math.floor(F.min())
maxF = math.ceil(F.max())
print(minF,maxF)

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
plt.title('Ackley function')
plt.show()
