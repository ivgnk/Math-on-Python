'''
on the base https://www.indusmic.com/post/ackleyn-2function
Mathematical Definition for 2D-Function
f(x,y) =
418.9829*2 - x * sin( sqrt( abs( x )))-y*sin(sqrt(abs(y)))

The function is usually evaluated on the hypercube xi ∈ [-500, 500], for all i = 1, …, d.
It has global minimum value  𝑓(𝑥∗)=0, at x∗ = (420.9687,…,420.9687)

https://yandex.ru/search/?clid=2353835&text=Schwefel+Function&lr=50
https://www.sfu.ca/~ssurjano/schwef.html
'''

import matplotlib.pyplot as plt
import numpy as np
import math

def f(x, y):
    return 418.9829*2 - x * np.sin( np.sqrt( abs( x )))-y*np.sin(np.sqrt(abs(y)))

X = np.linspace(-500,500)
Y = np.linspace(-500,500)

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
plt.title('Schwefel function')
plt.show()
