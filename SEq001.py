# Найти произведение xyz, если x, y, z - разные числа,
# удовлетворяющие системе уравнений
# Find the product xyz if x, y, z are different numbers,
# satisfying a system of equations

# x**2 + y**3 + z**3 = 0.8
# x**3 + y**2 + z**3 = 0.8
# x**3 + y**3 + z**2 = 0.8

import numpy as np
import scipy.optimize as optimize
from dataclasses import dataclass
from numba import njit

@njit
def calc():
    llst = []
    xs=-1e10; ys=-1e10; zs=-1e10
    min_ = 1e10
    for i in range(-2000,2001):
        # print(i,' ',min_,'  ',xs,'  ',ys,' ',zs)
        x = i/1000
        print(i)
        x2 = x*x
        x3 = x2*x
        for j in range(-2000, 2001):
            y = j/1000
            y2 = y * y
            y3 = y2*y
            for k in range(-2000, 2001):
                z = k/1000
                # print(x,' ',y,' ',z)
                z2 = z*z
                z3 = z2*z
                z_ = abs(x2 + y3 + z3 - 0.8) + abs(x3 + y2 + z3 - 0.8) + abs(x3 + y3 + z2 - 0.8)
                equal_ = (x == y) or (y == z) or (x == z)
                if equal_: z_ = z_*100
                if z_<min_:
                    min_ = z_; xs = x; ys = y; zs = z
                    llst.append([min_, xs, ys, zs])
                    print(z_)
    print(min_,'  ',xs,'  ',ys,' ',zs)
    print('1 equation = ',xs**2 + ys**3+ zs**3 - 0.8)
    print('2 equation = ',xs**3 + ys**2+ zs**3 - 0.8)
    print('3 equation = ',xs**3 + ys**3+ zs**2 - 0.8)
    # return z_, xs, ys, zs


# calc()
# Fun = 0.0005092720000001716
# x = 0.603    y = 0.604   z = 0.6

# calc2
# if minimization method = default then
# bad result, one of the local minima
# fun: 2.288888888914691
# x: [-3.333e-01 -3.333e-01 -3.333e-01]

# calc2
# if minimization method = 'Nelder-Mead' then
# good result
# fun: 3.2863840373820175e-05
# x:  [-2.442e-01 -2.441e-01  9.106e-01]

def calc2():
    def f(params):
        x, y, z = params
        z_ = abs(x**2 + y**3 + z**3 - 0.8) + abs(x**3 + y**2 + z**3 - 0.8) + abs(x**3 + y**3 + z**2 - 0.8)
        equal_ = (abs(x-y)<1e-8) or (abs(y-z)<1e-8) or (abs(x-z)<1e-8)
        if equal_: z_ = z_ * 10000
        return z_

    initial_guess = np.array([-2, -2, -2])
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    result = optimize.minimize(f, initial_guess, method='Nelder-Mead') # , tol=0.00001
    print(result)

calc2()


