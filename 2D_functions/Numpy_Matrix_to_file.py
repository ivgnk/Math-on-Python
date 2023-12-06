'''
on base of
https://numpy.org/doc/stable/reference/routines.io.html
https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html
https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html

https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html

https://numpy.org/doc/stable/user/how-to-io.html
'''
import numpy as np
import copy
import pprint
def output_1():
    x = np.arange(0.0,5.0,1.1)  # start, stop step
    y = copy.deepcopy(x)
    z = copy.deepcopy(x)
    fn1 = 'test1_1x.txt'; fn2 = 'test1_2xyz.txt'; fn3 = 'test1_3x_fmt.txt'; fn4 = 'test1_4.txt'
    np.savetxt(fname=fn1, X=x, delimiter=',')   # X is an array
    np.savetxt(fname=fn2, X=(x,y,z), delimiter='  ', fmt='%4.2f')   # x,y,z equal sized 1D arrays
    # np.savetxt('test1_x_fmt.txt', x, fmt='%1.4e')   # use exponential notation
    np.savetxt(fname=fn3, X=x, fmt='%4.2f')  # use fixed notation


    # https://pythonru.com/osnovy/fajly-v-python-vvod-vyvod
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile
    with open(fn4, 'w') as f:
        x.tofile(f, sep=' ', format="%s")

    return fn1, fn2, fn3, fn4

def input_1(fn1, fn2, fn3, fn4):
    # --1-- full 1D input
    x = np.loadtxt(fn1)
    print(x,'\nloadtxt')

    # --2-- full 2D input
    y2D = np.loadtxt(fn2)
    print(y2D,'\n')
    print(f'{y2D.shape=}')
    print('\nindexes')
    for i in range(y2D.shape[0]):
        print(i,' row',end=' ')
        for j in range(y2D.shape[1]):
            print((i,j),end=' ')
        print()
    # print('another after loadtxt')

    print('\nfull matrix input')
    for i in range(y2D.shape[0]):
        print(i,' row',end=' ')
        for j in range(y2D.shape[1]):
            print(y2D[i, j], end=' ')
        print()

    # --3-- partitial 2D input
    # https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
    print('\npartitial matrix input')
    y = np.loadtxt(fn2, usecols=(0, 2))
    print(y)

    # --4--
    print('\nfromfile vector')
    with open(fn4, 'r') as f:
        xx = np.fromfile(f, sep=' ')
    print(xx)
    print('\nfromfile matrix')
    with open(fn2, 'r') as f:
        xx = np.fromfile(f, sep=' ')
    print(xx); print(f'{xx.shape=}')
    print(f'{xx.dtype=}')
    # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    xx2 = np.reshape(xx, (3,5))  # 3 rows 5 columns
    print('\nafter reshape')
    print(xx2)

    # -- 4.1 --
    print('\nfromfile matrix with like')
    with open(fn2, 'r') as f:
        xx2 = np.fromfile(f, sep=' ',like=y2D)
    print(xx2); print(f'{xx2.shape=}')
    print(f'{xx2.dtype=}')


fn1, fn2, fn3, fn4 = output_1()
input_1(fn1, fn2, fn3, fn4)