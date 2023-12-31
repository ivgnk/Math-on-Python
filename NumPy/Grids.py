import numpy as np
import inspect
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels
from matplotlib.ticker import MultipleLocator
import warnings
from pprint import *

def for_square_matr2D_multiplication(m1:np.ndarray, m2:np.ndarray)->np.ndarray:
    # Only for 2D matrix
    # Алгоритм_Строка-на-столбец.png
    res:np.ndarray = np.array([[0 for x in range(2)] for y in range(2)])
    res[0,0] = m1[0,0]*m2[0,0]+m1[0,1]*m2[1,0]
    res[0,1] = m1[0,0]*m2[0,1]+m1[0,1]*m2[1,1]
    res[1,0] = m1[1,0]*m2[0,0]+m1[1,1]*m2[1,0]
    res[1,1] = m1[1,0]*m2[0,1]+m1[1,1]*m2[1,1]
    return res

def for_square_matr_multiplication(matrix1:np.ndarray, matrix2:np.ndarray, n:int)->np.ndarray:
    # www.geeksforgeeks.org/multiplication-two-matrices-single-line-using-numpy-python/
    res:np.ndarray = np.array([[0 for x in range(n)] for y in range(n)])

    # explicit for loops
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                # resulted matrix
                res[i][j] += matrix1[i][k] * matrix2[k][j]
    return res

def two_tests_for_2d_sq2_matrix():
    # compare
    # Алгоритм_Строка-на-столбец.png
    # and
    # www.geeksforgeeks.org/multiplication-two-matrices-single-line-using-numpy-python/

    a = np.array([[1, 0],
                  [0, 1]])
    b = np.array([[4, 1],
                  [2, 2]])
    c = np.matmul(a, b)
    # array([[4, 1],
    #        [2, 2]])
    print(c)

    res = for_square_matr2D_multiplication(a,b)
    print('Формульное умножение = \n',res)
    res = for_square_matr_multiplication(a, b, 2)
    print('Формульное умножение 2 = \n',res)

def thetests_for_2d_sq3_matrix():
    matrix1 = np.array([[1, 2, 3],
                        [3, 4, 5],
                        [7, 6, 4]])
    matrix2 = np.array([[5, 2, 6],
                        [5, 6, 7],
                        [7, 6, 4]])

    res = for_square_matr_multiplication(matrix1, matrix2, n=3)
    print('Формульное умножение\n',res,'\n')
    res = np.dot(matrix1, matrix2)
    print('np.dot умножение\n',res,'\n')
    res = matrix1 @ matrix2
    print('@ умножение\n',res,'\n')
    # res = np.outer(matrix1, matrix2)
    # print('np.outer умножение\n',res,'\n') # Возвращает: out(M, N) ndarray, out[i, j] = a[i] * b[j]
    res = np.matmul(matrix1, matrix2)
    print('np.matmul умножение\n',res,'\n')
    # [[36 32 32]
    #  [70 60 66]
    #  [93 74 100]]
    return res

def thetests_for_1d_3_matrix():
    # Алгоритм_Строка-на-столбец_для1D.png
    a1 = np.array([2, -5,  3])
    a2 = np.array([7,  0, -4])
    print('Two matrix a1, a2\n  a1=',a1,'\n a2 =', a2,'\n')
    res = np.dot(a1, a2)     # = 2
    print('np.dot умножение\n',res,'\n')
    res = a1*a2
    print('a1*a2 умножение (поэлементное)\n',res,'\n')

def imshow_visu_matr(matr:np.ndarray):
    # https://devpractice.ru/matplotlib-lesson-4-4-imshow-pcolormesh/
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    # --- 1
    plt.subplot(2, 2, 1)
    p1 = plt.imshow(matr,  aspect='equal', origin = 'upper') #
    plt.title("origin = 'upper'")
    fig.colorbar(p1)
    # --- 2
    plt.subplot(2, 2, 2)
    plt.title("origin = 'lower'")
    p2 = plt.imshow(matr, cmap = 'plasma', aspect='equal', origin = 'lower')
    fig.colorbar(p2)
    # --- 3
    plt.subplot(2, 2, 3)
    plt.title("origin = 'lower'")
    p3 = plt.imshow(matr, cmap = 'summer', aspect='equal', origin = 'lower',
                    extent=[0, 3, 0, 3] )
    ax[1,0].xaxis.set_major_locator(MultipleLocator(1))
    ax[1,0].yaxis.set_major_locator(MultipleLocator(1))
    # https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels
    # axs[2].xaxis.set_major_locator(MultipleLocator(1))
    plt.grid(which = 'major')
    fig.colorbar(p3)
    # --- 4
    plt.subplot(2, 2, 4)
    plt.title("origin = 'lower'")
    p4 = plt.imshow(matr, cmap = 'rainbow', aspect='equal', origin = 'lower',
                    extent=[0, 3, 0, 3] )
    ax[1,1].xaxis.set_major_locator(MultipleLocator(1))
    ax[1,1].yaxis.set_major_locator(MultipleLocator(1))
    plt.grid(which = 'major')
    fig.colorbar(p4)

    plt.show()

def the_meshgrid():
    print('\n', inspect.currentframe().f_code.co_name)
    # warnings.filterwarnings("ignore")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    x = np.arange(-5, 5, 0.5)
    y = np.arange(-5, 5, 0.5)
    xx, yy = np.meshgrid(x, y, sparse=True)
    try:
        z = np.sin(xx ** 2 + yy ** 2) / (xx ** 2 + yy ** 2)
    except Exception:
        print('some Error')
    print(f'{np.nanmin(z)=}    {np.nanmax(z)=} ')
    h = plt.contourf(x, y, z)
    fig.colorbar(h)
    plt.grid()
    plt.show()

def the_broadcast():
    # https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html
    x = np.array([[1], [2], [3]])
    y = np.array([4, 5, 6])
    b = np.broadcast(x, y)
    out = np.empty(b.shape)
    out.flat = [u+v for (u,v) in b]
    print(out)


def the_meshgrid2():
    print('\n', inspect.currentframe().f_code.co_name)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    x = np.arange(-5, 5, 1); xlen = len(x); print(f'{xlen=}')
    y = np.arange(-5, 5, 1); ylen = len(y); print(f'{ylen=}')
    xx, yy = np.meshgrid(x, y, sparse=True); xxshape = xx.shape; yyshape = yy.shape
    zz = xx+yy
    print(f'{xxshape=}');     print(f'{yyshape=}')
    # print(f'{zz=}')
    pp(zz)

    h = plt.contourf(x, y, zz)
    # plt.axis('scaled')
    plt.colorbar()
    plt.show()

    # z = x.+y
    # h = plt.contourf(x, y, z)
    # fig.colorbar(h)
    # plt.grid()
    # plt.show()

#---1
# tests_for_1d_3_matrix()
#---2
# res = tests_for_2d_sq3_matrix()
# imshow_visu_matr(res)

the_meshgrid2()

# x0 = np.linspace(-2, 2, 21)
# x1 = np.linspace(-2, 2, 21)
#
# lx0 = len(x0)
# lx1 = len(x1)
# z = np.zeros((lx1, lx0))
# print(z)

