"""
Яковлев И В
Дифференциальнеы уравнения
https://mathus.ru/phys/difur.pdf
стр.1-2
"""
from scipy.integrate import simps
from  sympy  import *

def var1():
    t = symbols('t')
    x = Function('x')
    m = symbols('m'); k = symbols('k'); a = symbols('a')
    q = symbols('q'); e0= symbols('e0'); w = symbols('w')
    # Объявление символьной функции
    f=m*x(t).diff(t,t)+k*x(t)+a*x(t).diff(t)-q*e0*sin(w*t)

    print('Исходное уравнение =\n',f) #
    f1 = dsolve(f,x(t))
    print('Решение уравнения =\n', f1)

def var2():
    t = symbols('t')
    x = Function('x')
    f=3*exp(t)+5*exp(2*t)
    f1=x(t).diff(t,t)-3*x(t).diff(t)+2*x(t)
    #----------------------------
    print('Первый путь v1 - Решение ДУ (дифференц. с помощью метода объекта)')
    print('Исходное уравнение =\n', f)  #
    f2 = dsolve(f1, x(t))
    print('Решение уравнения =\n', f2)
    print('Решение уравнения c раскрытыми скобками=\n', expand(f2))

    #----------------------------
    print('\nПервый путь v2 - Решение ДУ (дифференц. с помощью функции из пакета)')
    f3=diff(x(t),t,2)-3*diff(x(t),t)+2*x(t)
    f4 = dsolve(f3, x(t))
    print('Решение уравнения =\n', f4)
    print('Решение уравнения c раскрытыми скобками=\n', expand(f4))
    #----------------------------
    print('\nВторой путь - подстановка в уравнение')
    a1=diff(f,t); print('a1=', a1) # дифф 1 шаг - 1 произв.
    a2=diff(a1,t); print('a22=', a2) # дифф 2 шаг - 2 произв.
    a22=diff(f,t,2); print('a2=', expand(a22)) # дифф сразу
    print('Res3=',simplify(a2-3*a1+2*f)==0)
var2()
