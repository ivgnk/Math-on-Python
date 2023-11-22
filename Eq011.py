'''
https://www.youtube.com/watch?v=PtTJb9qyOys
1d function, 1d minimization
9**x + 9**x + 9**x = 999
solution = 1 + math.log(37,3)/2 = 2.6433995641091363
'''
import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.optimize as optimize
import math
from numba import njit

sign = lambda x: math.copysign(1, x)

minx, maxx = -100, 100
name_='function 2**x + x = 5'

@njit
def minf(x):
    return 2**x + x - 5

def minf2(x):
    if type(x) == float:
        d = abs(2**x + x - 5)
    else:
        d = np.abs(2**x + x - 5)
    # print(x,d)
    return d

def plot_func():
    n = 100
    xdat = np.linspace(-n, n, 201)
    ydat = minf(xdat)
    ydat2 = ydat - min(ydat)+0.00001
    ydat3 = minf2(ydat)
    for i,dat in enumerate(ydat):
        print(f'{i:4} {xdat[i]:5} {ydat[i]:12} {ydat2[i]:12} {ydat3[i]:12}')

    fig = plt.figure( figsize=(10,8))
    fig.suptitle(name_)
    plt.subplot(1, 2, 1)
    plt.plot(xdat,ydat)
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(xdat,ydat2)
    plt.semilogy()
    plt.grid()
    plt.show()

def desic1_enumeration():
    # Good
    resx, resy = None, None
    print(inspect.currentframe().f_code.co_name)
    global minx, maxx
    num = 1010000
    xdat = np.linspace(minx, maxx, num)
    ydat = np.linspace(minx, maxx, num)
    the_res=[]
    print('Перебор с выбором минимума')
    for i,x in enumerate(xdat):
        ydat[i] = abs(minf(x))
        if ydat[i] < 1e-3:
            print(f'{i} {x=} {ydat[i]}')
            resx = x
            resy = ydat[i]
    imin = np.argmin(ydat)
    print('Минимум')
    print(f' {imin=} {xdat[imin]} {ydat[imin]}')

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.title(name_)
    plt.plot(xdat,ydat)
    plt.xlim(minx, maxx)
    if not(resx is None): plt.plot(resx, resy,'ro')
    plt.grid()
    # https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    plt.show()

def desic2_least_squares():
    # Good
    print(inspect.currentframe().f_code.co_name)
    global minx, maxx
    initial_guess = np.array([maxx])
    # https://docs.scipy.org/doc/scipy/tutorial/optimize.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
    print('\noptimize.least_squares')
    result = optimize.least_squares(minf, initial_guess, bounds=optimize.Bounds(minx, maxx)) # , tol=0.00001
    print(result,'\n')

def desic3_minimize_scalar_bounded():
    # Good
    print(inspect.currentframe().f_code.co_name)
    global minx, maxx
    the_work=True; n =1
    while the_work:
        result = optimize.minimize_scalar(minf2, bounds=(minx, maxx), method='bounded', options={'maxiter':10000,'disp':True}) #
        print(result)
        if abs(result.fun) > 1e-6:
            ssign = sign(result.x)
            if ssign >=0:
                maxx = math.ceil(result.x)
            else:
                minx = math.floor(result.x)
            n += 1
            print(f'{n=} {minx=} {maxx=}')
            input()
        else:
            the_work = False

def desic4_minimize_scalar_golden():
    # Good
    print(inspect.currentframe().f_code.co_name)
    global minx, maxx
    the_work=True; n =1
    result = optimize.minimize_scalar(minf2, method='golden', options={'maxiter':10000,'disp':True}) #
    print(result)

def desic5_optimize_golden():
    # Good
    print(inspect.currentframe().f_code.co_name)
    global minx, maxx
    result = optimize.golden(minf2, brack=(minx, maxx), full_output=True, maxiter=100000)
    xmin, fval, funcalls = result
    print(f'{xmin=}   {fval=}')

def desic6_minimize_scalar():
    # Good
    print(inspect.currentframe().f_code.co_name)
    result = optimize.minimize_scalar(minf2)
    print(result)

def desic7_minimize_scalar_brent():
    #---- Good
    print(inspect.currentframe().f_code.co_name)
    result = optimize.minimize_scalar(minf2,method='brent')
    print(result)

def desic8_minimize_scalar_brent():
    #---- Good
    print(inspect.currentframe().f_code.co_name)
    global minx, maxx
    result = optimize.minimize_scalar(minf2,method='bounded',bounds=(minx, maxx))
    print(result)

def desic9_optimize_fminbound():
    #---- Good
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


if __name__ == "__main__":
    print(name_)
    # plot_func()
    s='Минимум imin=513663 1.715546253016086 0.00024403453041088596'

    n = 10
    match n:
        case 1: desic1_enumeration() # good
        case 2: desic2_least_squares() # good
        case 3: desic3_minimize_scalar_bounded() # good
        case 4: desic4_minimize_scalar_golden() # good
        case 5: desic5_optimize_golden() # good
        case 6: desic6_minimize_scalar() # good
        case 7: desic7_minimize_scalar_brent() # relative good
        case 8: desic8_minimize_scalar_brent() # good
        case 9: desic9_optimize_fminbound() # good
        case 10: desic10_optimize_root_scalar() # good
