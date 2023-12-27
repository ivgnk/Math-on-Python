import numpy as np
from copy import deepcopy
from pprint import *
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/35950050/how-to-import-python-file-located-in-same-subdirectory-in-a-pycharm-project
import pnumpy

# https://habr.com/ru/articles/484136/
from numba import njit
lst_func = []

# ----- Eq011



def for_eq011f(x):
    return 2**x + x - 5

def getEq011_f()->tuple:
    nnn = 100
    name_ = 'function 2**x + x = 5'
    return (nnn, for_eq011f, name_)

# ----- Eq012
def for_eq012f(x):
    return (3+x)/2022 + (2+x)/2023 + (1+x)/2024 + x/2025 + 4

def for_eq013f(x):
    if (type(x)==int) or (type(x)==float):
        if x == 0:
            return 1e50
        else:
            return 1 / x + 1 / (1 + x) - 1
    else:
        if x.any == 0:
            return 1e500
        else:
            return 1/x + 1/(1+x) - 1


def for_eq014f_pr(x):
    return (77**x - 121**x)


def for_eq014f(x):
    top =  (7**x - 11**x)
    bot = (77**x - 121**x)
    return top / (bot ** 0.5) - 1
    # if (type(bot)==int) or (type(bot)==float):
    #     if bot == 0:
    #         return None
    #     else:
    #         return top/(bot**0.5) - 1
    # else:
    #     if bot.any ==0:
    #         if bot == 0:
    #             return None
    #         else:
    #             return top/(bot**0.5) - 1



def getEq012_f()->tuple:
    nnn = 3300
    min_ = - nnn
    max_ = nnn
    name_ = 'function (3+x)/2022 + (2+x)/2023 + (1+x)/2024 + x/2025 = -4'
    return (nnn, min_,max_, for_eq013f, name_)

def getEq013_f()->tuple:
    nnn = 3300
    min_ = - nnn
    max_ = nnn
    name_ = 'function 1/x + 1/(1+x) = 1'
    return (nnn, min_,max_, for_eq013f, name_)

def getEq014_f()->tuple:
    nnn = 3300
    min_ = -50
    max_ = 50
    name_ = 'function (7**x - 11**x)/(77**x - 121**x)**0.5 - 1'
    return (nnn, min_,max_, for_eq014f, name_)

def getEq014_pr_f()->tuple:
    nnn = 3300
    min_ = -50
    max_ = 50
    name_ = 'function 77**x - 121**x'
    return (nnn, min_,max_, for_eq014f_pr, name_)

def for_eq015f(x):
    return (x**(2/3) - 9*x**(1/3)+8)

def getEq015_f()->tuple:
    nnn = 6300
    min_ = -600
    max_ = 600
    name_ = 'x**(2/3) - 9*x**(1/3)+8'
    return (nnn, min_,max_, for_eq015f, name_)

def for_eq015f_test():
    m = 1;    print(m,' ',for_eq015f(m) )
    m = 512;  print(m,' ',for_eq015f(m) )

def view_shape_n_size(title:str,z:np.ndarray)->None:
    print(title)
    print('size = ',z.size)
    print('shape = ',z.shape)
    print()


@njit
def for_eq016f(x0,x1):
    # on the base of
    # https://nlopt.readthedocs.io/en/latest/NLopt_Tutorial/#example-in-python
    # optimum at  0.33333309553516804 0.2962951203271458
    # minimum value =  0.5443299737541061
    # stackoverflow.com/questions/22774726/numpy-evaluate-function-on-a-grid-of-points
    lx1 = len(x1); lx0 = len(x0)
    Z = np.zeros((lx1, lx0))
    a1 = 2; b1 = 0; a2 = -1; b2 = 1
    # view_shape_n_size('ini Z param',Z)
    for ix0 in range(lx0):
        for jx1 in range(lx1):
            st1 = np.power((a1*x0[ix0]+b1),3)
            st2 = np.power(a2*x0[ix0]+b2,3)
            if (x1[jx1]>0) and (x1[jx1]>st1) and (x1[jx1]>st2):
                Z[ix0,jx1] = np.sqrt(x1[jx1]) + x1[jx1]-st1 + x1[jx1]-st2
            else:
                Z[ix0,jx1] = np.nan
    print('numbers of NaN = ',np.isnan(Z).sum(),'\n')
    # view_shape_n_size('end Z param',Z)
    ZZ = np.transpose(Z)
    return ZZ

def for_eq016f_test():
    # x = np.array([0,1,2,3]);  print(x**3)
    # optimum at  0.33333309553516804 0.2962951203271458
    # minimum value =  0.5443299737541061
    x0 = np.linspace(-0.5,1,3801)
    x1 = np.linspace(-1,8,3801)
    xd, yd = x0, x1
    print(xd)
    print('before calc')
    z = for_eq016f(xd, yd)
    print('after calc')
    view_shape_n_size('out Z param', z)

    the_minz = np.nanmin(z)
    the_maxz = np.nanmax(z)
    view_shape_n_size('out2 Z param', z)
    # https://stackoverflow.com/questions/2821072/is-there-a-better-way-of-making-numpy-argmin-ignore-nan-values
    the_miny_arg = np.nanargmin(z)
    the_maxy_arg = np.nanargmax(z)

    # s = np.unravel_index(np.nanmin(z), z.shape)
    smin = np.unravel_index(the_miny_arg, z.shape)
    smax = np.unravel_index(the_maxy_arg, z.shape)

    print('\nbuilt-in functions')
    print(' the_minz=',the_minz,' argmin(z)=',smin )
    print(' the_maxz=',the_maxz,' argmax(z)=',smax )

    print('\nmy functions')
    mmin, imin, jmin, mmax, imax, jmax = pnumpy.find_min_max_in_2Dmatr_with_nan(z)
    print(f'{mmin=}  {imin=}  {jmin=}')
    print(f'{mmax=}  {imax=}  {jmax=}')
    lvl = np.linspace(mmin, mmax, 11)
    plt.contour(xd, yd, z)
    curves2 = plt.contourf(xd, yd, z, lvl)
    curves3 = plt.contour(xd, yd, z, lvl, colors='k')
    plt.clabel(curves3, fontsize=10, colors='k')
    plt.grid()
    plt.show()


def create_function_list():
    # lst_func.append(getEq011_f())
    # lst_func.append(getEq012_f())
    # lst_func.append(getEq013_f())
    # lst_func.append(getEq014_pr_f())
    # lst_func.append(getEq014_f())
    lst_func.append(getEq015_f())
    print(lst_func)

if __name__ == "__main__":
    for_eq016f_test()

    # x0 = np.linspace(-2,2,801)
    # x1 = np.linspace(-4,4,801)
    # x = [x0,x1]
    # print(enumerate(x[0]))

    # x = np.array([[1, 2, np.nan, 4],
    #               [2, 3, np.nan, 5],
    #               [np.nan, 5, 2, 3]])
    # nn = np.isnan(x).sum()
    # print(f'{x.size=}')
    # print(f'{x.shape=}')
    # print('np.isnan(x).sum() = ',nn)

    # a = np.array([np.nan, 2.5, 3., np.nan, 4., 5.])  # 1
    # print(np.nanmin(a))
    # print(np.nanargmin(a))

