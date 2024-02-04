import numpy as np; import time
import random;  import numpy.random
import math

n = 0
maxx = 50_000
nelem = maxx*2
coo = [0.1]*nelem
t0 = time.time()
for i in range(nelem):
    coo[i] = math.sin(i)/math.cos(i)
    n+=1
print (time.time() - t0, "время обработки в секундах", 'n= ', n)

nn = n
arr = np.linspace(0.0, 2.0, nn)
n = 0
t0 = time.time()
for i in range(nelem):
    arr[i] = math.sin(i)/math.cos(i)
    n+=1
print (time.time() - t0, "время обработки в секундах", 'n= ', n)

