'''
https://svitla.com/blog/approximation-data-by-exponential-function-on-python
'''

#
# Как применить кусочно-линейную подгонку в Python?
# https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python
#
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# x = numpy.arange(1, 31, 1)
# y = numpy.array([3, 7, 14, 16, 26, 47, 73, 84, 113, 196, 218, 310, 356, 475, 548, 645, 794,
#                  942, 1096, 1251, 1319, 1462, 1668, 1892, 2203, 2511, 2777, 3102, 3372, 3764])

def func(x, a, b,c):
    return a * np.exp(-b * x)+c


# in arg> 40 then error fit
a1 = 5
b1 = 8
c1 = 5
st_  = 1
end_ = 15
x = np.arange(st_, end_, 1)
y = a1*np.exp(-b1*x)+c1
print(x)

[a, b, c], pcov = curve_fit(func, x, y)
print(a, b, c)

y1 = a * np.exp(b * x)+c

plt.plot(x, y, 'b', label = 'ini')
plt.plot(x, y1, 'r', label = 'appr', linestyle='--')
for yy, zz in zip(y, y1):
    print(f'{yy}  {zz}')

plt.legend()
plt.grid()
plt.show()