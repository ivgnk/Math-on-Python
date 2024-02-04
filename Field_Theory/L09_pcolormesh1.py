import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19_680_801);  data1 = np.random.randn(25, 25)
print(data1.min(),data1.max())

plt.figure(figsize=(13.5, 5)) # x и y – ширина и высота рис. в  дюймах
plt.subplot(1, 2, 1)
plt.pcolormesh(data1, cmap='plasma', edgecolors='face', shading='flat')
plt.colorbar(label='Value', drawedges=False)

plt.subplot(1, 2, 2)
plt.pcolormesh(data1,
               cmap='plasma',
               edgecolors='k',
               shading='gouraud',
               vmin=-2, vmax=1) # edgecolors не работает из-за shading='gouraud'
plt.colorbar(label='', drawedges=True )
plt.show()
