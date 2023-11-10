'''
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cspline1d.html#scipy.signal.cspline1d
'''
# Вычислите коэффициенты кубического сплайна для массива ранга 1.
# Найдите коэффициенты кубического сплайна для одномерного сигнала, предполагая зеркально-симметричные граничные условия.
# Чтобы получить сигнал обратно от сплайновое представление,
# зеркально-симметричное, свертка этих коэффициентов с длина 3 окна FIR [1.0, 4.0, 1.0]/6.0 .

# scipy.signal.cspline1d(signal, lamb=0.0)
# Parameters: signal(ndarray)
# A rank-1 array representing samples of a signal.
# lamb(float), optional - Smoothing coefficient, default is 0.0.
# c(ndarray) - Cubic spline coefficients.

import math
import numpy as np
import matplotlib.pyplot as plt
import inspect
from scipy.signal import cspline1d, cspline1d_eval

rng = np.random.default_rng()
sig1 = np.repeat([0., 1., 0.], 100)
print(sig1)
sig = sig1 + rng.standard_normal(len(sig1))*0.05  # add noise
time = np.linspace(0, len(sig))
filtered = cspline1d_eval(cspline1d(sig), time)
plt.plot(sig1, label="signal ini")
plt.plot(sig, label="signal with noise")
plt.plot(time, filtered, label="filtered")
plt.legend()
plt.show()
