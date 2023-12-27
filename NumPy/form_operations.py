import numpy as np
a = np.random.randint(5, size=(10, 3))
a[3,2] = -7
print('\n ini matr size=(10, 3)\n',a)

# https://numpy.org/doc/stable/reference/generated/numpy.transpose.html
print('after  np.transpose')
b = np.transpose(a)
print(b,'\n')

# https://stackoverflow.com/questions/3337301/numpy-matrix-to-array
print('after  np.array(a).flatten')
dd = np.array(a).flatten()
print(dd,'\n')

# https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
print('after  np.reshape(a, (3, 10))')
c = np.reshape(a, (3, 10))
print(c,'\n')

# min 1 var
print(f'\n {np.min(a)=}  {np.argmin(a)=} \n')

# min 2 var
# https://stackoverflow.com/questions/30180241/numpy-get-the-column-and-row-index-of-the-minimum-value-of-a-2d-array
s = np.unravel_index(a.argmin(), a.shape)
print('\n min 2 var -  \n',  s)

