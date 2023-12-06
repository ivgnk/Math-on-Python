'''
different ways for reading files
'''
import numpy as np

def output_1():
    irow=2; jcol = 3
    fn1 = 'test01_.txt'
    # https://numpy.org/doc/stable/user/basics.types.html
    a = np.zeros(shape=(irow, jcol), dtype=np.single)
    print(a)
    for i in range(irow):
        for j in range(jcol):
            a[i][j] = j+i*10
    print(a)
    np.savetxt(fname=fn1, X=a, delimiter=' ', fmt='%5.2f')  # X is an array

    x = np.loadtxt(fn1)
    print('x=\n',x)

    print('\nstrings in file')
    with open(fn1, 'r') as f:
        for line in f:
            print(line,end =' ')

    # www.geeksforgeeks.org/count-number-of-lines-in-a-text-file-in-python/

    with open(fn1, 'r') as fp:
        lines = len(fp.readlines())
    print('\nTotal Number of lines:', lines)

def output_2():
    n = 4
    a1 = [str(i)+'-'+str(i) for i in range(n)]
    print(f'{a1=}')
    a = np.array(a1)
    print(f'{a=}')

output_2()