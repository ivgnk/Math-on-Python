import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0.0, 5, 0.01)
y = np.cos(x*np.pi)
y_masked = np.ma.masked_where(y < -0.5, y)
plt.plot(x, y_masked)
plt.show()
