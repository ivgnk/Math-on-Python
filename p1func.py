# 1-D functions for testing in scipy.optimize

import matplotlib.pyplot as plt
import numpy as np

def funct_x3_1_(x):
    return x+x**2+x**3

def funct_x3_1(x, withview:bool=False):
    y = funct_x3_1_(x)
    if withview: plt.plot(x,y, label = 'x3_1');
    return y

def funct_x3_2(x, withview:bool=False):
    y = x-x**2+x**3
    if withview: plt.plot(x,y, label = 'x3_2');    return y

def funct_x3_3(x, withview:bool=False):
    y = -x-x**2+x**3
    if withview: plt.plot(x,y, label = 'x3_3');     return y

def funct_x3_4(x, withview:bool=False):
    y = -2*x-3*x**2+x**3
    if withview: plt.plot(x,y, label = 'x3_4');    return y

def funct_x3_5(x, withview:bool=False):
    y = -12*x-13*x**2+x**3
    if withview: plt.plot(x,y, label = 'x3_5');     return y

# ---- as_1d_rosenbrock
def as_1d_rosenbrock(x:np.ndarray):
    p1 = 100.0 * (1.4*x - x ** 2.0) ** 2.0
    p2 = 1*(1 - x) ** 2
    return p1, p2, p1+p2

def the_test_as_1d_rosenbrock():
    num_=140; edge = 1
    x = np.linspace(-edge, edge+1.4, num_)
    (yp1, yp2, yfull) = as_1d_rosenbrock(x)
    plt.plot(x,yp1,label = 'yp1')
    plt.plot(x,yp2,label = 'yp2')
    plt.plot(x,yfull,'r--',label = '1D rosenbrock')
    plt.legend()
    plt.show()