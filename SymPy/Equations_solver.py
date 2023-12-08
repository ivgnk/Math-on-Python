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

fun_2()