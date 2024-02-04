import random
import copy
import matplotlib.pyplot as plt
n = 100
name = [i for i in range(n)]
dat:list = [random.random() for i in range(n)]
# plt.plot(name, dat)
dat2:list = sorted(dat)
# plt.plot(name, dat2)
# plt.show()
dat3:list = copy.deepcopy(dat)
dat3.sort()
for i in range(n):
    print(i, dat3[i], dat2[i])
