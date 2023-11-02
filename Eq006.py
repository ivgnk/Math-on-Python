#  https://www.youtube.com/watch?v=CXCXY3h1iAk
# x**2 + (x**4 - x**2)**(1/3) = 2*x+1

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

num_=40
x = np.linspace(-10,10, num_)
def f(x):
    x2 = x**2
    return x2 + (x2*2 - x2)**(1/3) - (2*x+1)

y = f(x)

plt.subplot(1, 2, 1)
plt.plot(x,y)
plt.grid()

x2= np.linspace(-1,2, num_)
y2 = f(x2)

plt.subplot(1, 2, 2)
plt.plot(x2,y2)
plt.grid()
plt.show()
# for i in range(num_):
#     print(f' {i:3}   {x[i]:10}      {y[i]:10}')

print('------------- Calc2 -------------')
def calc2():
    def f(params):
        x = params
        x2 = x ** 2
        z_ = x2 + (x2*2 - x2)**(1/3) - (2*x+1)
        return z_

    initial_guess = np.array([-10])
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    result = optimize.minimize(f, initial_guess, method='Nelder-Mead') # , tol=0.00001
    print(result)

calc2()