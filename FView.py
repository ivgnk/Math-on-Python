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

def make_lines(dat:tuple, y:tuple):
    '''
    Делает две вертикальные линии для показа границ
    х координаты в dat, y координаты в ymin_, ymax_
    '''
    xleft = [dat[0], dat[0]]
    xrght = [dat[1], dat[1]]
    plt.plot(xleft, y, '--',color = 'k', label = 'gr1')
    plt.plot(xrght, y, '--',color = 'b', label = 'gr2')


