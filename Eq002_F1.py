# Решить уравнение X**X = 1 + ln(x)
# Solve the equation X**X = 1 + ln(x)
# https://www.youtube.com/watch?v=fjtq74SF-zQ

# Graph decision

import numpy as np
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


def eq2_graph(brcktsi=None):
    # ---- graphs decision
    if brcktsi is None:  brcktsi = [1, 30000]
    # brckts = [0.0001, 3]
    # https://ru.stackoverflow.com/questions/582684/range-и-вещественные-числа
    if brcktsi is None:
        brcktsi = [1, 30000]
    x1pr = [num / 10_000 for num in range(brcktsi[0], brcktsi[1] + 1, 200)]
    x1pr.append(1)
    x1pr.sort()
    # print(x1pr)
    x1: np.ndarray = np.array(x1pr)
    y1: np.ndarray = x1 ** x1
    y2: np.ndarray = 1 + np.log(x1)  # https://docs.python.org/3/library/math.html
    y3 = y2 - y1

    for i, y3d in enumerate(y3):
        if abs(y3d) <= 1e-10:
            print(f'{y3d=}  x ={x1[i]:8}  ')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(5))

    plt.plot(x1, y1, label='x**x')
    plt.plot(x1, y2, label='1+ln(x)')
    plt.plot(x1, y3, label='1+ln(x)-x**x', linewidth=5, color='red')

    x_l = [1, 1]
    y_l = [-7.5,10]
    plt.plot(x_l, y_l,label = 'x=1', color = 'green')

    plt.grid()
    plt.legend()
    plt.show() # показываем график

if __name__ == "__main__":
    eq2_graph()