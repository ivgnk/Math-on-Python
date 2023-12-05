'''
output_1() on base of
https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html
https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
'''
import numpy as np
import copy
import pprint
def output_1():
    x = np.arange(0.0,5.0,1.1)  # start, stop step
    y = copy.deepcopy(x)
    z = copy.deepcopy(x)
    fn1 = 'test1_1x.txt'; fn2 = 'test1_2xyz.txt'; fn3 = 'test1_3x_fmt.txt'
    np.savetxt(fname=fn1, X=x, delimiter=',')   # X is an array
    np.savetxt(fname=fn2, X=(x,y,z), delimiter='  ', fmt='%4.2f')   # x,y,z equal sized 1D arrays
    # np.savetxt('test1_x_fmt.txt', x, fmt='%1.4e')   # use exponential notation
    np.savetxt(fname=fn3, X=x, fmt='%4.2f')  # use fixed notation
    return fn1, fn2, fn3

def input_1(fn1, fn2, fn3):
    # --1-- full 1D input
    x = np.loadtxt(fn1)
    print(x,'\n')

    # --2-- full 2D input
    y = np.loadtxt(fn2)
    print(y,'\n')
    print(f'{y.shape=}')
    print('\nindexes')
    for i in range(y.shape[0]):
        print(i,' row',end=' ')
        for j in range(y.shape[1]):
            print((i,j),end=' ')
        print()

    print('\nmatrix')
    for i in range(y.shape[0]):
        print(i,' row',end=' ')
        for j in range(y.shape[1]):
            print(y[i, j], end=' ')
        print()

    # --3-- partitial 2D input
    # https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
    y = np.loadtxt(fn2)


fn1, fn2, fn3 = output_1()
input_1(fn1, fn2, fn3)