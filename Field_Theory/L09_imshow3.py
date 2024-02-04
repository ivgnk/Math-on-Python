import matplotlib.pyplot as plt
import numpy as np
i = 19_680_801  # запись с пробелами для удобства ввода
# инициализация так, чтобы у всех была одна картинка
#np.random.seed(i)
# выборка из стандартного нормального распределения
data1 = np.random.randn(25, 25)
plt.pcolor(data1 ,
  cmap=plt.get_cmap('viridis', 11))
# цветовая шкала
plt.colorbar()
plt.show()
