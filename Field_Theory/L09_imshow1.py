import matplotlib.pyplot as plt
import numpy as np
i = 19_680_801  # запись с пробелами для удобства ввода
# инициализация так, чтобы у всех была одна картинка
#np.random.seed(i)
# выборка из стандартного нормального распределения
data1 = np.random.randn(25, 25)
print(data1.min(), data1.max(), data1.mean())
# -3.6033264927121107 2.8010790020241485 0.06069894999450421
plt.imshow(data1)
plt.show()
