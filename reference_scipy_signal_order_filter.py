'''
Based on
https://docs.scipy.org/doc/scipy/reference/signal.html#filtering
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.order_filter.html#scipy.signal.order_filter

For 2-D and more dimensions
'''
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def reference_2D():
    x = np.arange(25).reshape(5, 5)
    domain = np.identity(3)
    print(f'\n{domain=}')
    print(x)
    # array([[ 0,  1,  2,  3,  4],
    #        [ 5,  6,  7,  8,  9],
    #        [10, 11, 12, 13, 14],
    #        [15, 16, 17, 18, 19],
    #        [20, 21, 22, 23, 24]])
    print(signal.order_filter(x, domain, 0))
    # array([[  0.,   0.,   0.,   0.,   0.],
    #        [  0.,   0.,   1.,   2.,   0.],
    #        [  0.,   5.,   6.,   7.,   0.],
    #        [  0.,  10.,  11.,  12.,   0.],
    #        [  0.,   0.,   0.,   0.,   0.]])
    print(signal.order_filter(x, domain, 2))
    # array([[  6.,   7.,   8.,   9.,   4.],
    #        [ 11.,  12.,  13.,  14.,   9.],
    #        [ 16.,  17.,  18.,  19.,  14.],
    #        [ 21.,  22.,  23.,  24.,  19.],
    #        [ 20.,  21.,  22.,  23.,  24.]])


if __name__ == "__main__":
    reference_2D()
