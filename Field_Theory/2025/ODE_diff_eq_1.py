"""
2022_Математика на Python_2_Криволапов_.pdf
стр.225
Дифференциальные уравнения первого порядка
"""
from sympy import *
from sympy.plotting import plot

# Объявление переменной
def var1():
    x = symbols('х')
    # Объявление символьной функции
    y = Function('у')
    # Уравнение, приведенное к нулевой правой части.
    # Функция записывается с указанием независимой переменной
    eq = diff(y(x),x) - (exp(sqrt(x)-2)/sqrt(x))
    # Решение уравнения для функции у(х)
    print(f'Дифференциальное уравнение:\n {eq}')
    print('Решение дифференциального уравнения\n',dsolve(eq, y(x)))
    print('Решение дифференциального уравнения с упрощением\n',dsolve(eq, y(x)).simplify())

# from sympy import *
# from sympy.plotting import plot
def var2():
    x = symbols('х')
    # Объявление символьной функции
    z = exp(sqrt(x)-2)/sqrt(x)
    print('Функция z\n', z)
    print('Результат интегрирования функции z\n', integrate(z, x))
    print('Результат интегрирования функции z с упрощением')
    inte=integrate(z, x).simplify()
    print(inte)
    # График
    plot( z, inte, title = 'Графики' , xlabel=' x ', ylabel=' y ' , legend = True)

from sympy import *
from sympy.plotting import plot
def du_rp():  # описание функции
    """
    ДУ с разделяющимися переменными
    2022_Математика на Python_2_Криволапов_.pdf
    с.226-227
    """
    x = symbols('x')
    y = Function('у')
    eq = (x + 1) * diff(y(x), x) + x * y(x)
    print('Уравнение\n',eq)
    print('Решение уравнения\n', dsolve(eq, y(x)))
    des=(x + 1)*exp(-x)
    plot(des, title='Частное решение уравнения', xlabel=' x ', ylabel=' y ', legend=True)

from sympy import *
from sympy.plotting import plot
def du_odn():  # описание функции
    x = symbols('x')
    y = Function('y')
    eq = x*diff(y(x),x) - y(x) - sqrt(y(x)**2-x**2)
    res=dsolve(eq, y(x))
    print('Уравнение\n',eq)
    print('Решение уравнения\n', res)
    # графики частных решений для y(x)=x*cosh(C1 - log(x)))
    desm = x * cosh(-2-log(x))
    des0=x*cosh(-log(x))
    desp = x*cosh(2-log(x))
    des_ = x * cosh(log(x)) # частное решение равное x * cosh(-log(x))
    # cosh - Чётная функция — функция, не изменяющая своего значения
    # при изменении знака независимой переменной
    # xlim=(-5, 5), ylim=(0, 1)
    plot(desm, des0,  title='Частные решения уравнения 1', xlabel=' x ', ylabel=' y ', legend=True)
    # xlim=(-5, 5), ylim=(-15, 15),
    plot(desp, des_,  title='Частные решения уравнения 2', xlabel=' x ', ylabel=' y ', legend=True)

du_odn()