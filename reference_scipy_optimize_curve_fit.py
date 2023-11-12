'''
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

xdata = np.linspace(0, 4, 50)
#a = 2.5; b = 1.3; c = 0.5 the_bounds = [5., 2., 1.5]
a = 5; b = 8; c = 5; the_bounds = [10.0, 10.0, 10.0]
y = func(xdata, a, b, c)
rng = np.random.default_rng()
y_noise = 0.2 * rng.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')

popt, pcov = curve_fit(func, xdata, ydata)
print(f'not bounds {popt=}')
# yfit= func(xdata, *popt)
yfit= func(xdata, popt[0], popt[1], popt[2])
plt.plot(xdata, yfit, 'r--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt)+' not bounds',linewidth=3)

# Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, the_bounds))
print(f'with bounds {popt=}')

plt.plot(xdata, func(xdata, *popt), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt)+' with bounds')

plt.title('scipy.optimize.curve_fit')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
