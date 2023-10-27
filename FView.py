import numpy as np
import matplotlib.pyplot as plt

def calc_arr(bounds:list[int],n:int, divisor:int): # mult - коэфф. для перевода в действительные
    x = np.linspace(bounds[0],bounds[1],n)/divisor
    return x

def view_f1D(x, f):
    y = f(x)
    plt.plot(x, y)

    plt.grid()
    # plt.legend()
    plt.show() # показываем график
