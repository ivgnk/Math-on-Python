'''
https://www.youtube.com/watch?v=5N-lYlVsYY4
1d function, 1d minimization
x**2 - 5 = sqrt(x+5)

sol1 = (-1-math.sqrt(17))/2 = -2.5615528128088303
sol2 = (-1+math.sqrt(21))/2 = 1.79128784747792
'''
import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.optimize as optimize
import math

sign = lambda x: math.copysign(1, x)
minx, maxx = -5, 15

def minf(x):
    return (x**2 - 5) - math.sqrt(x+5)

def minf2(x):
    d = abs(x**2 - 5 - np.sqrt(x+5))
    # print(x,d)
    return d

def desic1_enumeration():
    # Good
    global minx, maxx
    resx, resy = None, None
    print(inspect.currentframe().f_code.co_name)
    minx, maxx, nn = -5, 15, 160000
    xdat = np.linspace(minx, maxx, nn)
    ydat = np.linspace(minx, maxx, nn)
    for i,x in enumerate(xdat):
        res = minf(x)
        # print(i,x,res)
        ydat[i] = res
        if abs(res) < 4e-4:
            print(f'{x=} {res=}')
            resx = x
            resy = res
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title('y = (x**2 - 5) - math.sqrt(x+5)')
    plt.plot(xdat,ydat)
    # plt.xlim(minx, maxx)
    # plt.ylim(-20, 20)
    if not(resx is None): plt.plot(resx, resy,'ro')
    plt.grid()
    # https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.show()

def desic2_least_squares():
    # Good но нашли 1 корень
    print(inspect.currentframe().f_code.co_name)
    global minx, maxx
    initial_guess = np.array([minx])
    # https://docs.scipy.org/doc/scipy/tutorial/optimize.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
    print('\noptimize.least_squares')
    result = optimize.least_squares(minf, initial_guess, bounds=optimize.Bounds(minx, maxx)) # , tol=0.00001
    print(result,'\n')

def desic3_minimize_scalar_bounded():
    # разделил интервал на 2 части по числу корней
    print(inspect.currentframe().f_code.co_name)
    global minx, maxx

    print('1 root')
    maxx1 = 0
    result = optimize.minimize_scalar(minf2, bounds=(minx, maxx1), method='bounded',
                                      options={'maxiter': 10000, 'disp': True})  #
    print(result)
    print('\n2 root')
    minx1 = 0
    result = optimize.minimize_scalar(minf2, bounds=(minx1, maxx), method='bounded',
                                      options={'maxiter': 10000, 'disp': True})  #
    print(result)

    # while the_work:
    #     result = optimize.minimize_scalar(minf2, bounds=(minx, maxx), method='bounded', options={'maxiter':10000,'disp':True}) #
    #     print(result)
    #     if abs(result.fun) > 1e-5:
    #         ssign = sign(result.x)
    #         if ssign >=0:
    #             maxx = math.ceil(result.x)
    #         else:
    #             minx = math.floor(result.x)
    #         n += 1
    #         print(f'{n=} {minx=} {maxx=}')
    #         print('Print any key')
    #         input()
    #     else:
    #         the_work = False

def desic4_minimize_scalar_golden():
    #---- Не работает и нет параметра bounds, чтобы перезаустить для суженого диапазона поиска
    print(inspect.currentframe().f_code.co_name)
    global minx, maxx
    the_work=True; n =1
    result = optimize.minimize_scalar(minf2, method='golden', options={'maxiter':10000,'disp':True}) #
    print(result)
    #---- Не работает и нет параметра bounds, чтобы перезаустить для суженого диапазона поиска

def desic5_optimize_golden():
    # Не работает
    print(inspect.currentframe().f_code.co_name)
    global minx, maxx
    result = optimize.golden(minf2, brack=(minx, maxx), full_output=True)
    xmin, fval, funcalls = result
    print(f'{xmin=}   {fval=}')

def desic6_minimize_scalar():
    #---- Не работает и нет параметра bounds, чтобы перезаустить для суженого диапазона поиска
    print(inspect.currentframe().f_code.co_name)
    result = optimize.minimize_scalar(minf2)
    print(result)

def desic7_minimize_scalar_brent():
    #---- Не работает и нет параметра bounds, чтобы перезаустить для суженого диапазона поиска
    print(inspect.currentframe().f_code.co_name)
    result = optimize.minimize_scalar(minf2,method='brent')
    print(result)

def desic8_minimize_scalar_brent():
    #---- Не работает как надо c широкими пределами
    print(inspect.currentframe().f_code.co_name)
    global minx, maxx
    print(f'{minx=} {maxx=}')
    result = optimize.minimize_scalar(minf2,method='bounded',bounds=(minx, maxx))
    print(result)

def desic9_optimize_fminbound():
    #---- Не работает как надо c широкими пределами
    # ---- Работает как надо c узкими пределами
    # Univariate (scalar) minimization methods
    print(inspect.currentframe().f_code.co_name)
    global minx, maxx
    result = optimize.fminbound(minf2,minx, maxx, full_output=True)
    minimizer, fval, ierr, numfunc = result
    print(f'{minimizer=} {fval=} {ierr=} {numfunc=}')

def desic10_optimize_root_scalar():
    # Good
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html#scipy.optimize.root_scalar
    print(inspect.currentframe().f_code.co_name)
    global minx, maxx
    # print(minf(minx), minf(maxx))
    print('\nbrentq')
    sol = optimize.root_scalar(minf, bracket=(minx, maxx), method='brentq')
    print(f'{sol.root=}  {minf(sol.root)=}')

    print('\nbrenth')
    sol = optimize.root_scalar(minf, bracket=(minx, maxx), method='brenth')
    print(f'{sol.root=}  {minf(sol.root)=}')

    print('\nbisect')
    sol = optimize.root_scalar(minf, bracket=(minx, maxx), method='bisect')
    print(f'{sol.root=}  {minf(sol.root)=}')

    print('\nridder')
    sol = optimize.root_scalar(minf, bracket=(minx, maxx), method='ridder')
    print(f'{sol.root=}  {minf(sol.root)=}')

    s='toms748'
    print('\n'+s)
    sol = optimize.root_scalar(minf, bracket=(minx, maxx), method=s)
    print(f'{sol.root=}  {minf(sol.root)=}')


desic1_enumeration()