'''
Calculation and visualization of the inclined plane
'''

import matplotlib.pyplot as plt
import numpy as np
import pprint

fn1 = 'slope_function_dat.txt'
def f(x, y):
    return x+y


X = np.linspace(0,6, num=7)  # 50 points by default
Y = np.linspace(0,6, num=7)  # 50 points by default
print(f'{X=}')

# lvl

x, y = np.meshgrid(X,Y)
F = f(x,y)
# https://stackoverflow.com/questions/1939228/constructing-a-python-set-from-a-numpy-matrix
# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
F_lvl=list(set(F.flatten()))
print(f'{F_lvl=}')
np.savetxt(fname=fn1, X=F, delimiter=' ', fmt='%7.2f')  # X is an array


fig =plt.figure(figsize=(8,8))
# see also Eq_2D_002.py
curves1 = plt.contourf(x,y,F,F_lvl) # 40 - num isolines
# curves2 = plt.contour(x,y,F,F_lvl) # 40 - num isolines
curves3 = plt.contour(x, y, F, F_lvl, colors='k')
plt.clabel(curves3, fontsize=10, colors='k')

fig.colorbar(curves1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('inclined plane')
plt.show()

# --- data structure
print('\ndata structure info')
print(type(F)) #

print(f'{X.size=}   {Y.size=}')
print(f'{F.size=}') # 2500
print(f'{F.shape=}') # (50, 50)

print('\nF = ')
pprint.pprint(F)
print('\nF[0] = ')
pprint.pprint(F[0])

# for i, xi in enumerate(X):
#     for j, yj in enumerate(Y):
#         print(f'{i:2} {j:2} {F[i,j]}   {f(xi,yj)}')
#         input()