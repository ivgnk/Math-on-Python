import numpy as np
a = np.random.randint(5, size=(10, 3))
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

