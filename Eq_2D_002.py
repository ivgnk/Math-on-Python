'''
Solve the equation
1 = 16**(x**2+y) +16**(x+y**2)
https://www.youtube.com/watch?v=gezN0drkOMc
'''
import pprint
import inspect
import copy
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def func1(x,y):
    return x**2-y-31

@njit
def func2(x,y):
    return y**2-x-31

@njit
def wrk():
    lst_res = []
    n = 30
    step = 0.001
    prec=1e-02
    data = (1e10, 1e10)
    xmin, ymin = None, None
    x = np.arange(-n, n+step, step)
    y = np.arange(-n, n+step, step)
    for xi in x:
        for yj in y:
            res0 = func1(xi,yj)
            res1 = func2(xi,yj)
            if (abs(res0) < prec) and (abs(res1) < prec):
                    lst_res.append((res0,res1, round(xi,3), round(yj,3)))

    print('All results ')
    for i,x in enumerate(lst_res):
        print(i,' res0=',x[0],' res1=',x[1],x[2],x[3], func1(x[2], x[3]),func2(x[2], x[3]))
    print('\nlst_res')
    return lst_res


def calc_for_plot():
    n = 20
    step = 0.025
    lst =[]; lst_name=[]
    x = np.arange(-n, n+step, step)
    y = np.arange(-n, n+step, step)
    z = np.outer(x, y)
    # -1 var
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            z[i,j] = func1(xi,yj)+func2(xi,yj)
    lst.append(copy.deepcopy(z))
    lst_name.append('f1+f2')
    print('1',np.min(z),np.max(z))

    # -2 var
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            z[i,j] = func1(xi,yj)-func2(xi,yj)
    lst.append(copy.deepcopy(z))
    lst_name.append('f1-f2')
    print('2',np.min(z),np.max(z))

    # -3 var
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            z[i,j] = abs(func1(xi,yj))+abs(func2(xi,yj))
    lst.append(copy.deepcopy(z))
    lst_name.append('abs(f1)+abs(f2)')
    print('3',np.min(z),np.max(z))

    # -4 var
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            z[i,j] = abs(func1(xi,yj))-abs(func2(xi,yj))
    lst.append(copy.deepcopy(z))
    lst_name.append('abs(f1)-abs(f2)')
    print('4',np.min(z),np.max(z))

    nx = 2
    ny = 2
    return x,y,lst, nx, ny, lst_name


def the_plot(x,y,f,nx:int,ny:int, lst_name:list):
    # for xi in x:
    #     for yj in y:
    #         res0 = func1(xi,yj)
    #         res1 = func2(xi,yj)
    # func =
    print('\n',inspect.currentframe().f_code.co_name)
    fig = plt.figure(figsize=(10, 8))
    plt.suptitle('Variants of functions')
    n = 0
    for i in range(nx):
        for j in range(ny):
            z = f[n]
            mmin = int(round(np.min(z),0))
            mmax = int(round(np.max(z),0))
            print(n,mmin,mmax)
            lvl = np.linspace(mmin, mmax, 9)
            plt.subplot(nx,ny,n+1)
            curves2 = plt.contourf(x, y, z,lvl)
            fig.colorbar(curves2)
            plt.contour(x, y, z,lvl)
            curves3 = plt.contour(x, y, z, lvl, colors='k')
            plt.title('n='+str(n)+', '+str(mmin)+' '+str(mmax)+',    '+lst_name[n])
            plt.clabel(curves3, fontsize=10, colors='k')
            n +=1
    plt.show()


# lst_res = wrk()
# pprint.pprint(lst_res)
x,y,lst,nx,ny, lst_name = calc_for_plot()
the_plot(x,y,lst,nx,ny, lst_name)