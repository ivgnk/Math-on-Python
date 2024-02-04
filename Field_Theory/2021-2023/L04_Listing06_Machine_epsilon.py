import numpy as np

print(np.finfo(float).eps)
# вывод     2.220446049250313e-16
print(np.finfo(np.float32).eps)
# вывод      1.1920929e-07

eps=1
while (eps+eps/2)>eps:
    eps = eps/2
print(eps)