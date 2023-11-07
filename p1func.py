# 1-D functions for testing in scipy.optimize

import matplotlib.pyplot as plt

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
