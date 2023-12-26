import numpy as np
# Get the number of nonzero elements in a numpy array
# stackoverflow.com/questions/12451954/get-the-number-of-nonzero-elements-in-a-numpy-array
x = np.array([[1, 2, np.nan, 4],
              [2, 3, np.nan, 5],
              [np.nan, 5, 2, 3]])
nn = np.isnan(x).sum()
print(nn)

# Is there a better way of making numpy.argmin() ignore NaN values
# https://stackoverflow.com/questions/2821072/is-there-a-better-way-of-making-numpy-argmin-ignore-nan-values
a = np.array([np.nan, 2.5, 3., np.nan, 4., 5.])  # 1
print(np.nanmin(a))
print(np.nanargmin(a))
