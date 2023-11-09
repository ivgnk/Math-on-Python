'''
Based on
https://docs.scipy.org/doc/scipy/reference/signal.html#filtering

Filter data along one-dimension with an IIR or FIR filter
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html#scipy.signal.lfilter
'''
# import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def ref_1D_lfilter():
    '''
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html#scipy.signal.lfilter
    '''
    # --(1)------------ Generate a noisy signal to be filtered:
    import numpy as np
    from scipy import signal
    import matplotlib.pyplot as plt
    rng = np.random.default_rng()
    t = np.linspace(-1, 1, 201)
    x = (np.sin(2 * np.pi * 0.75 * t * (1 - t) + 2.1) +
         0.1 * np.sin(2 * np.pi * 1.25 * t + 1) +
         0.18 * np.cos(2 * np.pi * 3.85 * t))
    xn = x + rng.standard_normal(len(t)) * 0.08

    plt.plot(x,label='ini');     plt.plot(xn,label='with noise')
    plt.grid(); plt.legend() ;    plt.show()

    # --(2)------------ Filtering
    # Create an order 3 lowpass butterworth filter:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    b, a = signal.butter(3, 0.05)
    # signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
    # N - Порядок фильтра.
    # Wn - Критическая частота или частоты. Для фильтров нижних и верхних частот Wn является скаляром;
    # для полосовых и полосовых фильтров Wn представляет собой последовательность длины 2
    # Для фильтра Баттерворта это точка, в которой коэффициент усиления падает до
    # 1/sqrt(2) = 0.7071067811865475 от коэффициента полосы пропускания («точка -3 дБ»).

    # Apply the filter to xn. Use lfilter_zi to choose the initial condition of the filter
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, xn, zi=zi * xn[0])
    # Apply the filter again, to have a result filtered at an order the same as filtfilt:
    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
    # Use filtfilt to apply the filter:
    y = signal.filtfilt(b, a, xn)

    # --(2)------------ Plotting
    # Plot the original signal and the various filtered versions:
    # plt.figure
    plt.plot(t, xn, 'b', alpha=0.75)
    plt.plot(t, z, 'r--', t, z2, 'r', t, y, 'k')
    plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice',
                'filtfilt'), loc='best')
    plt.plot(t, x, label='ini');
    plt.grid(True)
    plt.show()


ref_1D_lfilter()
# my_1D_wienerfilt()
