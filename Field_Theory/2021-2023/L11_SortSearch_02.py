import random
import copy
import numpy as np
import matplotlib.pyplot as plt
n = 100
name = [i for i in range(n)]
dat:list = [random.random() for i in range(n)]
dat_np = np.array(dat)
dat2_np = np.array(dat)
dat2_np.sort()
dat3_np = sorted(dat_np)
for i in range(n):
    print(i, dat_np[i], dat2_np[i], dat3_np[i], dat2_np[i]-dat3_np[i])