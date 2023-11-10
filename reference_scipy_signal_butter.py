'''
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
'''
# signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
# N - Порядок фильтра.

# Wn - Критическая частота или частоты. Для фильтров нижних и верхних частот Wn является скаляром;
# для полосовых и полосовых фильтров Wn представляет собой последовательность длины 2
# Для фильтра Баттерворта это точка, в которой коэффициент усиления падает до
# 1/sqrt(2) = 0.7071067811865475 от коэффициента полосы пропускания («точка -3 дБ»).

# btype {'lowpass', 'highpass', 'bandpass', 'bandstop'}, необязательно
# Тип фильтра. По умолчанию используется «низкочастотный».

# analog=False
# Если True, возвращает аналоговый фильтр, в противном случае возвращается цифровой фильтр.

# output='ba'
# Тип вывода: числитель/знаменатель («ba»), полюс-ноль («zpk») или секции второго порядка («sos») (Second-order sections).
# По умолчанию используется «ba» для обратной совместимости, но для фильтрации общего назначения
# следует использовать «sos».

# fs - Частота дискретизации цифровой системы.
# Возвращает :
# b, a (ndarray, ndarray) - Полиномы числителя ( b ) и знаменателя ( a ) БИХ-фильтра.
# Возвращается только в том случае output='ba'.
#
# z, p, k (ndarray, ndarray, float)
# Нули, полюса и системное усиление передаточной функции БИХ-фильтра. Возвращается только в том случае output='zpk'.
#
# sos (ndarray, ndarray)
# Представление секций второго порядка БИХ-фильтра. Возвращается только в том случае output='sos'.

import math
import numpy as np
import matplotlib.pyplot as plt
import inspect
from scipy import signal

def ref_1_analog_filter():
    from scipy import signal
    import matplotlib.pyplot as plt
    import numpy as np
    b, a = signal.butter(4, 100, 'low', analog=True)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqs.html
    # signal.freqs returns: w (ndarray) - The angular frequencies at which h was computed.
    # h (ndarray) - The frequency response
    w, h = signal.freqs(b, a)
    fig = plt.figure(figsize=(14,7))
    fig.suptitle("Butterworth filter frequency response", fontsize=16)
    ax1 = plt.subplot(1, 2, 1)
    plt.title('Standard plot')
    plt.plot(w, abs(h))
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency [radians / second]')
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green')  # cutoff frequency # Add a vertical line across the Axes
    plt.axhline(1/math.sqrt(2), color='red')

    ax2 = plt.subplot(1, 2, 2)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Semilogx plot and log10(abs(h))')
    plt.ylabel('Amplitude [dB]')
    plt.xlabel('Frequency [radians / second]')
    # plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green')  # cutoff frequency # Add a vertical line across the Axes
    plt.axhline(20*math.log10(1/math.sqrt(2)), color='red')
    plt.show()

def sig_gen():
    print(inspect.currentframe().f_code.co_name)

    # --(1)-- Generate a signal made up of 10 Hz and 20 Hz, sampled at 1 kHz
    t = np.linspace(0, 1, 1000, False)  # 1 second
    sig = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, sig)
    ax1.set_title('10 Hz and 20 Hz sinusoids')
    ax1.axis([0, 1, -2, 2])

    # --(2)--
    # Design a digital high-pass filter at 15 Hz to remove the 10 Hz tone, and apply it to the signal.
    # (It’s recommended to use second-order sections format when filtering,
    # to avoid numerical error with transfer function (ba) format):
    sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
    filtered = signal.sosfilt(sos, sig)
    ax2.plot(t, filtered)
    ax2.set_title('After 15 Hz high-pass filter')
    ax2.axis([0, 1, -2, 2])
    ax2.set_xlabel('Time [seconds]')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ref_1_analog_filter()
    sig_gen()