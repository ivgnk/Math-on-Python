'''
https://en.wikipedia.org/wiki/Himmelblau%27s_function
https://www.indusmic.com/post/himmelblau-function

All posts in
https://www.indusmic.com/blog/page/11

Wiki:
f(x,y)=(x**2+y-11)**2+(x+y**2-7)**2

'''

import matplotlib.pyplot as plt
import numpy as np
import pprint


def f(x, y):
    return (((x ** 2 + y - 11) ** 2) + (((x + y ** 2 - 7) ** 2)))


X = np.linspace(-6,6, num=20)  # 50 points by default
Y = np.linspace(-6,6, num=20)  # 50 points by default

x, y = np.meshgrid(X,Y)
F = f(x,y)

fig =plt.figure(figsize=(9,9))

ax=plt.axes(projection='3d')
ax.contour3D(x,y,F,40) # 40 - num isolines

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('F')
ax.set_title('Himmelblau Function')
ax.view_init(50,50)
plt.show()

# --- data structure
print(type(F)) #

print(f'{X.size=}   {Y.size=}')
print(f'{F.size=}') # 2500
print(f'{F.shape=}') # (50, 50)

pprint.pprint(F)
pprint.pprint(F[0])
for i, xi in enumerate(X):
    for j, yj in enumerate(Y):
        print(f'{i:2} {j:2} {F[i,j]}   {f(xi,yj)}')
        input()