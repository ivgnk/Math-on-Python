'''
https://stackoverflow.com/questions/tagged/numpy
stackoverflow.com/questions/12451954/get-the-number-of-nonzero-elements-in-a-numpy-array
'''
import numpy as np
row = 100
col = 100
a = np.random.randint(0, 3, size=(row,col))
print(a)
b = len(a.nonzero()[0])
print(b)
c = np.count_nonzero(a)
print(c)
ssum = 0
for i in range(row):
    ssum += np.count_nonzero(a[i])
print(ssum)