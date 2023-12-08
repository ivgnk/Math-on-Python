'''
Linear equations solving
https://docs.sympy.org/latest/modules/solvers/solvers.html
'''

from sympy.solvers import solve
from sympy import Symbol
from pprint import pp

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

def fun_6():
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




fun_6()