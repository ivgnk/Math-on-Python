'''
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

xdata = np.linspace(0, 4, 50)
a = 2.5; b = 1.3; c = 0.5
y = func(xdata, a, b, c)
rng = np.random.default_rng()
y_noise = 0.2 * rng.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')

popt, pcov = curve_fit(func, xdata, ydata)
print(f'{popt=}')
# yfit= func(xdata, *popt)
yfit= func(xdata, popt[0], popt[1], popt[2])
plt.plot(xdata, yfit, 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt)+' not bounds')

# Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))

plt.plot(xdata, func(xdata, *popt), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt)+' with bounds')

plt.title('scipy.optimize.curve_fit')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
