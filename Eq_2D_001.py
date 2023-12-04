'''
Solve the equation
1 = 16**(x**2+y) +16**(x+y**2)
https://www.youtube.com/watch?v=gezN0drkOMc
'''

def func(x,y):
    return 16**(x**2+y) +16**(x+y**2) - 1

import numpy as np
n = 10
step = 0.1
data = 1e10
xmin, ymin = None, None
x = np.arange(-n, n+step, step)
y = np.arange(-n, n+step, step)
for xi in x:
    for yj in y:
        res = func(xi,yj)
        if data > res:
            data = res
            xmin, ymin = xi, yj

print(f'{data=}  {xmin=}  {ymin=}')
xmin2= round(xmin,2)
ymin2= round(ymin,2)
print(f'{data=}  {xmin2=}  {ymin2=}')
print('Test result')
print(func(xmin2,ymin2))
