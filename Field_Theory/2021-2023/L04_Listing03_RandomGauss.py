import random
import matplotlib.pyplot as plt
n = 2000
x = [i for i in range(n)]
yy = [2, 1, 0.5]
dat = []
for i in range(len(yy)):
    sps = [random.gauss(0, yy[i]) for j in range(n)]
    dat.append(sps)
    plt.plot(x, dat[i], label ='s='+str(yy[i]) )

plt.legend(loc='upper right')
plt.grid()
plt.show()
