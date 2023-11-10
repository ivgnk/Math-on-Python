'''
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html#scipy.signal.detrend
Remove linear trend along axis from data.
'''

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

rng = np.random.default_rng()
npoints = 1000
noise = rng.standard_normal(npoints)
x1 = 3 + 2*np.linspace(0, 1, npoints)
x = x1 + noise
y = signal.detrend(x)

as_line = x -y

fig = plt.figure(figsize=(14, 7))
plt.plot(noise,label='noise')
plt.plot(x,label='line+noise')
plt.plot(x1,label='line')
plt.plot(y,label='detrend', linestyle='dotted')
plt.plot(as_line,label='as_line', linestyle='dotted')
plt.legend(); plt.grid()
plt.show()
print('max diff ini and detrend data',(signal.detrend(x) - noise).max())
diff = (signal.detrend(x) - noise)/noise; diff_pr=np.abs(diff*100)
n = 15
for n, d, d1 in zip(noise, diff, diff_pr):
    print(f'{n:15}  {d:15}   {d1:15}')
print('max diff ini and detrend data in %',diff_pr.max())
print('mean diff ini and detrend data in %',diff_pr.mean())
print('median diff ini and detrend data in %',np.median(diff_pr))