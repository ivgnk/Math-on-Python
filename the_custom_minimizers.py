# Based on https://docs.scipy.org/doc/scipy/tutorial/optimize.html#custom-minimizers

# Sometimes, it may be useful to use a custom method as a (multivariate or univariate) minimizer,
# for example, when using some library wrappers of minimize (e.g., basinhopping).

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import inspect

from scipy.optimize import OptimizeResult
from copy import deepcopy
from p1func import *

def custom_multivariate_minimization():
    def custmin(fun, x0, args=(), maxfev=None, stepsize=0.1,
            maxiter=100, callback=None, **options):
        bestx = x0
        besty = fun(x0)
        funcalls = 1
        niter = 0
        improved = True
        stop = False

        while improved and not stop and niter < maxiter:
            improved = False
            niter += 1
            for dim in range(np.size(x0)):
                for s in [bestx[dim] - stepsize, bestx[dim] + stepsize]:
                    testx = np.copy(bestx)
                    testx[dim] = s
                    testy = fun(testx, *args)
                    funcalls += 1
                    if testy < besty:
                        besty = testy
                        bestx = testx
                        improved = True
                if callback is not None:
                    callback(bestx)
                if maxfev is not None and funcalls >= maxfev:
                    stop = True
                    break

        return OptimizeResult(fun=besty, x=bestx, nit=niter,
                              nfev=funcalls, success=(niter > 1))

    print(inspect.currentframe().f_code.co_name)


    # ---- minimisation
    x0 = np.array([1.35, 0.9, 0.8, 1.1, 1.2])
    res = optimize.minimize(optimize.rosen, x0, method=custmin, options=dict(stepsize=0.05))
    print(res)


def custom_univariate_minimization():
    def custmin(fun, bracket, args=(), maxfev=None, stepsize=0.1,
                maxiter=100, callback=None, **options):
        bestx = (bracket[1] + bracket[0]) / 2.0
        besty = fun(bestx)
        funcalls = 1
        niter = 0
        improved = True
        stop = False

        while improved and not stop and niter < maxiter:
            improved = False
            niter += 1
            for testx in [bestx - stepsize, bestx + stepsize]:
                testy = fun(testx, *args)
                funcalls += 1
                if testy < besty:
                    besty = testy
                    bestx = testx
                    improved = True
            if callback is not None:
                callback(bestx)
            if maxfev is not None and funcalls >= maxfev:
                stop = True
                break

        return OptimizeResult(fun=besty, x=bestx, nit=niter,
                              nfev=funcalls, success=(niter > 1))

    def f(x):
        return (x - 2) ** 2 * (x + 2) ** 2

    print(inspect.currentframe().f_code.co_name)
    # ---- min  imisation
    br = (1, 2)
    # tr_br= (-3.5, 0)
    f2 = f
    f2 = as_1d_rosenbrock2
    res = optimize.minimize_scalar(f2, bracket=br, method=custmin,
                          options=dict(stepsize=0.05))
    print(res)  # x = -2.0
    # ---- visualisation
    num_=140; edge = 1
    x = np.linspace(br[0], br[1], num_)
    y = f2(x)

    xres2 = res.x
    yres2 = res.fun

    plt.plot(x,y,label= 'curve')
    plt.plot(xres2, yres2, 'bo', label='min 1')

    plt.legend()
    plt.grid()
    plt.show()

custom_univariate_minimization()