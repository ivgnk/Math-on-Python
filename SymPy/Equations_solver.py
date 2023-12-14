'''
Linear equations solving
https://docs.sympy.org/latest/modules/solvers/solvers.html
'''

from sympy.solvers import solve
from sympy import Symbol
from pprint import pp
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
f1 = Symbol('f1')
f2 = Symbol('f2')
f3 = Symbol('f3')

def fun_1():
    print(f'{solve(x**2 - 1, x) = }')   # [-1, 1]
    print(f'{solve(x ** 2 - 4) = }')            # [-2, 2]
    print(f'{solve([x - 3, y**2 - 1]) =}')      # [{x: 3, y: -1}, {x: 3, y: 1}]

def fun_2():
    '''
     https://www.youtube.com/watch?v=z7-p3tfjNS8
     x**3 - xyz = -5
     y**3 - xyz = 2
     z**3 - xyz = 21
    '''
    def f1c(x,y,z)->float:
        return x**3 - x*y*z + 5

    def f2c(x,y,z)->float:
        return y**3 - x*y*z -2

    def f3c(x,y,z)->float:
        return z**3 - x*y*z -21

    f1 = x**3 - x*y*z + 5
    f2 = y**3 - x*y*z - 2
    f3 = z**3 - x*y*z - 21
    # print(f'{solve([f1, f2, f3]) =}')  # [{x: 3, y: -1}, {x: 3, y: 1}]
    lst = solve([f1, f2, f3])
    # pp(lst)
    n = len(lst)
    print(f'\n{n=}')

    exact_res = []
    # testing result for 1 equation
    tol = 1e-14
    for i in range(n):
        the_sl = lst[i]
        print('\n',i)
        print(the_sl) # type(the_sl)
        print('x=', the_sl[x],'  y=', the_sl[y],'  z=', the_sl[z])
        # print('x=', type(the_sl[x]), '  y=', type(the_sl[y]), '  z=', type(the_sl[z]))
        x1 = complex(the_sl[x])
        y1 = complex(the_sl[y])
        z1 = complex(the_sl[z])
        print(f'{x1=}    {y1=}    {z1=}')
        resf1 = f1c(x1,y1,z1)
        resf2 = f2c(x1,y1,z1)
        resf3 = f3c(x1,y1,z1)
        mod_res = abs(resf1) + abs(resf2) + abs(resf3)
        if mod_res < tol: exact_res.append(i)
        print(f'{resf1= }  {resf2= }  ')
        print(f'{resf3= }  {mod_res = }')

    print(f'\nWith sum of modules < {tol}')
    print(exact_res,' from '+str(n))

def fun_3():
    '''
    https://www.youtube.com/watch?v=H_Zqk7Esl7g
    4*x**4-16*x**3+3*x**2+4*x-1=0
    '''
    x = Symbol('x')
    print(f'{solve(4*x**4-16*x**3+3*x**2+4*x-1, x) = }')   # [-1/2, 1/2, 2 - sqrt(3), sqrt(3) + 2]

def fun_4():
    '''
    https://www.youtube.com/watch?v=Dx9qkJXd9OE
    (2*x**2-3*x+4)**(1/2) - (2*x**2+x+3)**(1/2)=1-4*x
    '''
    x = Symbol('x')
    print(f'{solve((2*x**2-3*x+4)**(1/2) - (2*x**2+x+3)**(1/2)-1+4*x, x) = }') # [0.25]

def fun_5():
    '''
    https://www.youtube.com/watch?v=2g5I0H0JMSQ
    (x+6)!/(x+2)! -1680
    https://ru.wikipedia.org/wiki/Факториал
    https://ru.wikipedia.org/wiki/Факториал#Связь_с_гамма-функцией
    '''
    x = Symbol('x')
    # f1= Symbol('(x+3)*(x+4)*(x+5)*(x+6)-1680')
    f1 = (x+3)*(x+4)*(x+5)*(x+6)-1680
    res_lst = solve(f1, x)
    print(f'All results {res_lst = }') # [-11, 2, -9/2 - sqrt(159)*I/2, -9/2 + sqrt(159)*I/2]
    n = len(res_lst);
    good_lst = []
    # выделение нужногоответа - неотрицательного числа
    for i in range(n):
        zz = complex(res_lst[i])
        if (zz.imag ==0) and (zz.real>0):
            good_lst.append(zz.real)
    print('True results')
    print(good_lst)    # [2]

def fun_6_not_sympy():
    '''
    https://www.youtube.com/watch?v=PxTaSH9tPEw
    ( 2**(1/2))**(x**(1/2)) + 2**(1/2)**(y**(1/2) ) = 504
    x and y are positive integers
    '''
    from math import sqrt, pow
    good_res = []
    nn = 1100
    dat = sqrt(2)
    for i in range(0,nn):
        for j in range(0, nn):
            d1 = pow(dat,sqrt(i))
            d2 = pow(dat,sqrt(j))
            f = d1 - d2 - 504
            # print(i,' ',j,'  ',d1,'  ',d2,'  ',f)
            if abs(f) <1e-10: good_res.append([i,j,f])
    n = len(good_res)
    print(f'{n=}')
    print('True results')
    print(good_res)
    print(f'{sqrt(good_res[0][0]) = }  {sqrt(good_res[0][1]) = }')
    print('Test = ',2**9-2**3)

def fun_7_not_sympy():
    '''
    x**5 = 9**x
    https://www.youtube.com/watch?v=tOPVVOjgx5w
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    good_res = []
    nn = 1.5
    dat = np.linspace(-nn,nn, round(145*(nn+1)))
    fres = np.linspace(-nn, nn, round(145 * (nn + 1)))
    for i,x in enumerate(dat):
        f = x**5 - 9**x
        fres[i] = f
        if abs(f) < 1e-1:
            print(i,'  ',i**5,'  ',9**i)

    plt.plot(dat,fres)
    plt.grid()
    plt.show()
    print('test =',0**5 - 9**0)

def fun_7_with_sympy():
    '''
    x**5 = 9**x
    https://www.youtube.com/watch?v=tOPVVOjgx5w
    '''
    x = Symbol('x')
    f1 = x**5 - 9**x
    res_lst = solve(f1, x)
    print(f'All results {res_lst = }') #


def fun_8():
    '''
    x**x = 100
    https://www.youtube.com/watch?v=vieSDhMFJPM
    '''
    x = Symbol('x')
    f1 = x**x - 100
    res_lst = solve(f1, x)
    print(f'All results {res_lst = }') #
    print(complex(res_lst[0]))
    r = float(res_lst[0])
    print(r,' ',r**r)

def fun_8_not_sympy():
    '''
    x**x = 100
    https://www.youtube.com/watch?v=vieSDhMFJPM
    '''

    import numpy as np
    import matplotlib.pyplot as plt

    good_res = []
    nn = 5
    dat = np.linspace(0,nn, round(11145*(nn+1)))
    fres = np.linspace(0, nn, round(11145 * (nn + 1)))
    for i,x in enumerate(dat):
        f = x**x - 100
        fres[i] = f
        if abs(f) < 0.01:
            print(i,'  ',x,'  ',f)

    plt.plot(dat,fres)
    plt.grid()
    plt.show()
    # print('test =',0**5 - 9**0)

def fun_9():  ## BAD Function
    '''
    (7-4*sqrt(3))**(x+sqrt(x+2)) = (2-sqrt(3))**((2*x+4)**(1/(x+3)))
    https://www.youtube.com/watch?v=6HNjc5MzwCc
    '''
    from math import sqrt
    x = Symbol('x')
    dat1 = 7-4*(sqrt(3))
    dat2 = 2-sqrt(3)
    f1 = dat1**(x+(x+2)**0.5) -dat2**((2*x+4)**(x+1/3))
    res_lst = solve(f1, x)
    print(f'All results {res_lst = }') #

def fun_9_not_sympy(): ## BAD Function
    '''
    (7-4*sqrt(3))**(x+sqrt(x+2)) = (2-sqrt(3))**((2*x+4)**(1/(x+3)))
    https://www.youtube.com/watch?v=6HNjc5MzwCc
    '''
    good_res = []
    nn = 5
    dat = np.linspace(-2,nn, round(11145*(nn+1)))
    fres = np.linspace(-2, nn, round(11145 * (nn + 1)))
    for i,x in enumerate(dat):
        f = (7-4*sqrt(3))**(x+sqrt(x+2)) - (2-sqrt(3))**((2*x+4)**(x+1/3))
        fres[i] = f
        if abs(f) < 1e-5:
            print(i,'  ',x,'  ',f)

    plt.plot(dat,fres)
    plt.grid()
    plt.show()
    # print('test =',0**5 - 9**0)

def fun_10():
    '''
    x**2+y**2 = 31
    x**3+y**3 = 154
    https://www.youtube.com/watch?v=09nYaqupTjk
    '''
    f1 = x**2 + y**2 - 31
    f2 = x**3 + y**3 - 154
    lst = solve([f1, f2])
    # pp(lst)
    n = len(lst)
    print(f'\n{n=}')
    print(lst)

def fun_11():  ## BAD - NotImplementedError: could not solve y**(5 - y) + (5 - y)**y - 17
    '''
    x**y + y**x = 17
    x+y = 5
    https://www.youtube.com/watch?v=9ivsSVbWyfQ
    decision x = 2, y = 3
    '''
    x = Symbol('x')
    y = Symbol('y')
    f1 = x**y + y**x - 17
    f2 = x + y - 5
    # lst = solve([f1, f2])
    lst = solve([x**y + y**x - 17, x + y - 5])
    # pp(lst)
    n = len(lst)
    print(f'\n{n=}')
    print(lst)

def fun_12():
    '''
    https://www.youtube.com/watch?v=Qrvt2HT-W28
    4**(x**2-x-6) = 7
    '''
    lst = solve([4**(x**2-x-6) - 7])
    n = len(lst)
    print(f'\n{n=}')
    print(lst)

def fun_13():
    '''
    3**(x**2) = 27
    '''
    lst = solve([3**(x**2) - 27])
    n = len(lst)
    print(f'\n{n=}')
    print(lst)

def fun_14():  ## BAD
    '''
    9**y - 6**y = 4**y
    https://www.youtube.com/watch?v=UmnBh5wk0iU
    '''
    x = Symbol('x')
    lst = solve([9**x - 6**x - 4**x])
    n = len(lst)
    print(f'\n{n=}')
    print(lst)

def fun_15():
    '''
    x**6 - x**3 = 2
    https://www.youtube.com/watch?v=cwZbT2BgFxc
    '''
    lst = solve([x**6 - x**3 - 2])
    n = len(lst)
    print(f'\n{n=}')
    print(lst)

def fun_16():
    '''
    https://www.youtube.com/watch?v=V3NBSQCCSlQ
    sqrt(x-2) = 5
    (x-2)**0.5 = x+3
    x**(1/3) - x = 1
    '''
    lst = solve([(x-2)**0.5 - 5]);     print(lst)
    lst = solve([(x-2)**0.5 - x+3]);    print(lst)
    lst = solve([x**(1/3) - x - 1]);    print(lst)
    lst = solve([(x+1)*3 - x]);    print(lst)
    lst = solve([3**(1/2)*x - 2**(1/2)]);    print(lst)
    # - - -
    lst = solve([(15-2*x)**(1/2) - 3]);    print(lst)


def fun_16_not_sympy():
    '''
    x**(1/3) - x = 1
    '''
    good_res = []
    nn = 5
    dat = np.linspace(0,nn, round(11145*(nn+1)))
    fres = np.linspace(0, nn, round(11145 * (nn + 1)))
    for i,x in enumerate(dat):
        f = x**(1/3) - x - 1
        fres[i] = f
        if abs(f) < 1e-5:
            print(i,'  ',x,'  ',f)

    plt.plot(dat,fres)
    plt.grid()
    plt.show()

def fun_17():
    '''
    https://www.youtube.com/watch?v=w1Vl63siU3s
    x**(2/3) - 9*x**(1/3)+8
    x = 1
    x = 512
    '''
    lst = solve([x**(2/3) - 9*x**(1/3)+8]);     print(lst)

def fun_18():
    '''
    https://www.youtube.com/watch?v=orDjrxeTP2A
    sqrt(x-1) + sqrt(2*x-3) + sqrt(3*x-5) + sqrt(4*x-7) -5*x+6
    :return:
    '''
    # lst = solve([((x-1)**0.5 + (2*x-3)**0.5 + (3*x-5)**0.5 + (4*x-7)**0.5)**2 - (5*x + 6)**2 ])
    lst = solve([(x-1)**0.5 + (2*x-3)**0.5 + (3*x-5)**0.5 + (4*x-7)**0.5 + 5*x - 6])
    print(lst)

fun_18()
# fun_16_not_sympy()