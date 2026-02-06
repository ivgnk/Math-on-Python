# import matplotlib.pyplot as plt
# import numpy as np
# # (https://labex.io/ru/tutorials/matplotlib-2d-image-plotting-with-pcolormesh-48860)
#
# Z = np.random.rand(6, 10)
# x = np.arange(-0.5, 10, 1)
# y = np.arange(4.5, 11, 1)
#
# fig, ax = plt.subplots()
# ax.pcolormesh(x, y, Z)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Создаём данные для pcolormesh
x = np.linspace(20, 30, 20)
y = np.linspace(-5, 5, 15)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Строим цветовой mesh
plt.pcolormesh(X, Y, Z, shading='auto')


# Задаём метки осей
plt.xlabel('Ось X (единицы измерения)')
plt.ylabel('Ось Y (единицы измерения)')

plt.title('Пример pcolormesh с подписями осей')
plt.colorbar(label='Значение Z')  # подпись для цветовой шкалы
plt.show()
