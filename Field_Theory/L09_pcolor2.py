import matplotlib.pyplot as plt
import numpy as np
i = 19_680_801  # запись с пробелами для удобства ввода
# инициализация так, чтобы у всех была одна картинка
#np.random.seed(i)
# выборка из стандартного нормального распределения
data1 = np.random.randn(25, 25)
plt.figure(figsize=(13.5, 5)) # x и y – ширина и высота рис. в  дюймах
plt.subplot(1, 2, 1)
plt.title('pcolor')
plt.pcolor(data1); plt.colorbar()
plt.subplot(1, 2, 2)
plt.title('imshow')
plt.imshow(data1);  plt.colorbar()
plt.show()
