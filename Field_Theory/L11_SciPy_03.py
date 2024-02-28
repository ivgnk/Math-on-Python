'''
SciPy FFT
'''
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

np.random.seed(42)
num = 300 # число отсчетов
t = np.arange(0,num,1)
# https://thispointer.com/how-to-generate-random-array-in-numpy/
sig = np.random.normal(loc=0, scale=1, size=num)
sig_t = sig+t*0.01

plt.figure(figsize=(12,6))
plt.suptitle('Trend & detrend data')
sig_d = signal.detrend(sig_t)
plt.subplot(121)
plt.plot(t,sig,label ='ini')
plt.plot(t,sig_t,label ='ini+trend'); plt.legend(); plt.grid()
plt.subplot(122)
plt.plot(t,sig,label ='ini')
plt.plot(t,sig_d, linestyle='-',label ='detrend'); plt.legend(); plt.grid()
plt.show()

rel_err = (sig-sig_d)/sig*100 # относительная погрешность-1
abs_err = sig-sig_d # абсолютная погрешность
rel_err2 = np.abs(abs_err/sig*100) # относительная погрешность-2

print('     sig     sig_d   abs_err    rel_err    rel_err2')
for i in range(0,num):
    print(f'{sig[i]:8.4f}  {sig_d[i]:8.4f}  {abs_err[i]:8.4f}  {rel_err[i]:8.4f}  {rel_err2[i]:8.4f}')

print()
print(f'{np.min(sig     )=}    {np.max(sig     )=}')
print(f'{np.min(abs_err )=}    {np.max(abs_err )=}')
print()
print(f'{np.min(rel_err)=}    {np.max(rel_err)=}')
print(f'{np.min(rel_err2)=}   {np.max(rel_err2)=}')
print()
print(f'{np.average(rel_err)=}    {np.average(rel_err)=}')
print(f'{np.median(rel_err)=}    {np.median(rel_err)=}')
print()
print(f'{np.average(rel_err2)=}    {np.average(rel_err2)=}')
print(f'{np.median(rel_err2)=}    {np.median(rel_err2)=}')

# print(f'{np.argmin(res2   )=}    {np.argmax(res2   )=}')
