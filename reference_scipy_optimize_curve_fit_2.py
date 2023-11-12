'''
https://svitla.com/blog/approximation-data-by-exponential-function-on-python
'''

#
# Как применить кусочно-линейную подгонку в Python?
# https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python
#
import numpy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# x = numpy.arange(1, 31, 1)
# y = numpy.array([3, 7, 14, 16, 26, 47, 73, 84, 113, 196, 218, 310, 356, 475, 548, 645, 794,
#                  942, 1096, 1251, 1319, 1462, 1668, 1892, 2203, 2511, 2777, 3102, 3372, 3764])

# in arg> 40 then error fit
x = numpy.arange(1, 35, 1)
y = 0.1*numpy.exp(0.08*x)
print(y)

[a, b], res1 = curve_fit(lambda x1, a, b: a * numpy.exp(b * x1), x, y)

y1 = a * numpy.exp(b * x)

plt.plot(x, y, 'b', label = 'ini')
plt.plot(x, y1, 'r', label = 'appr', linestyle='--')
plt.legend()
plt.grid()
plt.show()