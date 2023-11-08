'''
Based on
https://docs.scipy.org/doc/scipy/reference/signal.html#filtering
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.medfilt.html#scipy.signal.medfilt
'''
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def my_1D_medfilt():
    # https://numpy.org/doc/stable/reference/random/index.html
    rng = np.random.default_rng() # Generator start
    x = rng.standard_normal(250)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(x,label='ini')
    lst = [5, 15, 25, 35]  # range(5,30,8)
    for i in lst:
        print(i)
        y = signal.medfilt(x, i)
        plt.plot(y,label='medfilt '+str(i))
    # y = signal.medfilt(x, 15)
    # plt.plot(y)
    plt.legend()
    plt.grid()
    plt.show()

my_1D_medfilt()
