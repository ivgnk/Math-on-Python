import matplotlib.pyplot as plt
import numpy as np
i = 19_680_801  # запись с пробелами для удобства ввода
# инициализация так, чтобы у всех была одна картинка
#np.random.seed(i)
# выборка из стандартного нормального распределения
data1 = np.random.randn(25, 25)
# потому что горизонтальный colorbar
# сжимает по вертикали
plt.figure(figsize=(5, 7)) # ширина и высота
plt.pcolor(data1, cmap=plt.get_cmap('magma', 6));
plt.colorbar(orientation='horizontal', # ориентация шкалы
             ticks=[-2, -1, 0, 1,  2], # метки шкалы
             label='Value', # название шкалы
             drawedges=True # края цветовых градаций
             )
plt.show()
