'''
Module with different methods of 1D minimization
'''
import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.optimize as optimize
import math
import types
from p1D_func_for_mininimizations import lst_func, create_function_list
from numba import njit

sign = lambda x: math.copysign(1, x)
'''
Examples of minf(x) and minf2(x) functions 
def minf(x):
    return (3+x)/2022 + (2+x)/2023 + (1+x)/2024 + x/2025 + 4

'''

fff:types.FunctionType

def minf2(x):
    global fff
    d = np.abs(fff(x))
    return d

def plot_func(n:int, minx:float, maxx:float, minf:types.FunctionType, name_:str)->None:
    print('\n',inspect.currentframe().f_code.co_name)
    global fff;  fff = minf
    xdat = np.linspace(minx, maxx, n)
    ydat = minf(xdat)
    ydat2 = ydat - min(ydat)+0.00001
    ydat3 = minf2(ydat)
    for i,dat in enumerate(ydat):
        print(f'{i:4} {xdat[i]:5} {ydat[i]:12} {ydat2[i]:12} {ydat3[i]:12}')

    fig = plt.figure(figsize=(10,8))
    fig.suptitle(name_)
    plt.subplot(1, 2, 1)
    plt.plot(xdat,ydat)
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(xdat,ydat2)
    plt.semilogy()
    plt.grid()
    plt.show()

def desic1_enumeration(minx:float, maxx:float, minf:types.FunctionType, name_:str):
    resx, resy = None, None
    print('\n',inspect.currentframe().f_code.co_name)
    num = 1010000
    xdat = np.linspace(minx, maxx, num)
    ydat = np.linspace(minx, maxx, num)
    the_res=[]
    print('Перебор с выбором минимума')
    for i,x in enumerate(xdat):
        ydat[i] = abs(minf(x))
        if (not math.isnan(ydat[i])) and (ydat[i] < 1e-3):
            print(f'{i} {x=} {ydat[i]}')
            resx = x
            resy = ydat[i]
    imin = np.argmin(ydat)
    print('Минимум ',inspect.currentframe().f_code.co_name)
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

def desic2_least_squares(minx:float, maxx:float,minf:types.FunctionType):
    print('\n',inspect.currentframe().f_code.co_name)
    initial_guess = np.array([maxx])
    # https://docs.scipy.org/doc/scipy/tutorial/optimize.html
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
    print('\noptimize.least_squares')
    try:
        result = optimize.least_squares(minf, initial_guess, bounds=optimize.Bounds(minx, maxx)) # , tol=0.00001
    except Exception:
        print('Error in ',inspect.currentframe().f_code.co_name)
        result = None
    print(f'{result=} \n')


def desic3_minimize_scalar_bounded(minx:float, maxx:float, minf:types.FunctionType):
    print('\n',inspect.currentframe().f_code.co_name)
    global fff;  fff = minf
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


def desic4_minimize_scalar_golden(minx:float, maxx:float, minf:types.FunctionType):
    print('\n',inspect.currentframe().f_code.co_name)
    global fff;  fff = minf
    the_work=True; n =1
    try:
        result = optimize.minimize_scalar(minf2, method='golden', options={'maxiter':10000,'disp':True}) #
    except Exception:
        print('Error in ',inspect.currentframe().f_code.co_name)
        result = None
    print(f'{result=}')


def desic5_optimize_golden(minx:float, maxx:float, minf:types.FunctionType):
    print('\n',inspect.currentframe().f_code.co_name)
    global fff;  fff = minf
    try:
        result = optimize.golden(minf2, brack=(minx, maxx), full_output=True, maxiter=100000)
        xmin, fval, funcalls = result
        print(f'{xmin=}   {fval=}')
    except Exception:
        print('Error in ',inspect.currentframe().f_code.co_name)
        result = None


def desic6_minimize_scalar(minf:types.FunctionType):
    global fff;  fff = minf
    print('\n',inspect.currentframe().f_code.co_name)
    try:
        result = optimize.minimize_scalar(minf2)
    except Exception:
        print('Error in ',inspect.currentframe().f_code.co_name)
        result = None
    print(f'{result=}')

def desic7_minimize_scalar_brent(minf:types.FunctionType):
    global fff;  fff = minf
    print('\n',inspect.currentframe().f_code.co_name)
    try:
        result = optimize.minimize_scalar(minf2,method='brent')
    except Exception:
        print('Error in ',inspect.currentframe().f_code.co_name)
        result = None
    print(f'{result=}')

def desic8_minimize_scalar_brent(minx:float, maxx:float, minf:types.FunctionType):
    global fff;  fff = minf
    print('\n',inspect.currentframe().f_code.co_name)
    result = optimize.minimize_scalar(minf2,method='bounded',bounds=(minx, maxx))
    print(result)

def desic9_optimize_fminbound(minx:float, maxx:float, minf:types.FunctionType):
    global fff;  fff = minf
    # Univariate (scalar) minimization methods
    print('\n',inspect.currentframe().f_code.co_name)
    result = optimize.fminbound(minf2,minx, maxx, full_output=True)
    minimizer, fval, ierr, numfunc = result
    print(f'{minimizer=} {fval=} {ierr=} {numfunc=}')

def desic10_optimize_root_scalar(minx:float, maxx:float, minf:types.FunctionType):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html#scipy.optimize.root_scalar
    print('\n title for many function = ',inspect.currentframe().f_code.co_name)
    # print(minf(minx), minf(maxx))
    print('\nbrentq')
    try:
        sol = optimize.root_scalar(minf, bracket=(minx, maxx), method='brentq')
        print(f'{sol.root=}  {minf(sol.root)=}')
    except Exception:
        print('Error in ',inspect.currentframe().f_code.co_name)

    print('\nbrenth')
    try:
        sol = optimize.root_scalar(minf, bracket=(minx, maxx), method='brenth')
        print(f'{sol.root=}  {minf(sol.root)=}')
    except Exception:
        print('Error in ',inspect.currentframe().f_code.co_name)

    print('\nbisect')
    try:
        sol = optimize.root_scalar(minf, bracket=(minx, maxx), method='bisect')
        print(f'{sol.root=}  {minf(sol.root)=}')
    except Exception:
        print('Error in ',inspect.currentframe().f_code.co_name)

    print('\nridder')
    try:
        sol = optimize.root_scalar(minf, bracket=(minx, maxx), method='ridder')
        print(f'{sol.root=}  {minf(sol.root)=}')
    except Exception:
        print('Error in ',inspect.currentframe().f_code.co_name)

    s='toms748'
    print('\n'+s)
    try:
        sol = optimize.root_scalar(minf, bracket=(minx, maxx), method=s)
        print(f'{sol.root=}  {minf(sol.root)=}')
    except Exception:
        print('Error in ',inspect.currentframe().f_code.co_name)

def all_minimizations_testing(n:int, minx:float, maxx:float, minf:types.FunctionType, name_:str)->None:
    print(name_)
    for i in range(11):
        match i:
            case 0: plot_func(n, minx, maxx, minf, name_)
            case 1: desic1_enumeration(minx, maxx, minf, name_)
            case 2: desic2_least_squares(minx, maxx, minf)
            case 3: desic3_minimize_scalar_bounded(minx, maxx, minf)
            case 4: desic4_minimize_scalar_golden(minx, maxx, minf)
            case 5: desic5_optimize_golden(minx, maxx, minf)
            case 6: desic6_minimize_scalar(minf)
            case 7: desic7_minimize_scalar_brent(minf)
            case 8: desic8_minimize_scalar_brent(minx, maxx, minf)
            case 9: desic9_optimize_fminbound(minx, maxx, minf)
            case 10: desic10_optimize_root_scalar(minx, maxx, minf)

if __name__ == "__main__":
    # lst_func =[]
    # def minf__(x):
    #     return (3 + x) / 2022 + (2 + x) / 2023 + (1 + x) / 2024 + x / 2025 + 4
    # nnn = 2300
    # lst_func.append((nnn,minf__,'(3 + x) / 2022 + (2 + x) / 2023 + (1 + x) / 2024 + x / 2025 + 4'))
    # print(lst_func)
    create_function_list()

   # for i in range(len(lst_func)):
    for i in range(1):
        nnn_ = lst_func[i][0]
        min_ = lst_func[i][1]
        max_ = lst_func[i][2]
        minf__1 = lst_func[i][3]
        name1 = lst_func[i][4]
        all_minimizations_testing(n = nnn_, minx=min_, maxx=max_, minf = minf__1, name_= name1)