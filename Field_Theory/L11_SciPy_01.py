'''
Ввод и вывод в матрицы Matlab
'''

import numpy as np
from scipy import io as spio

a = np.ones((3,3))
print(a)
spio.savemat('file.mat',{'a':a}) # сохраняем матрицу как элемент словаря
data = spio.loadmat('file.mat')
print(data) # вывод всей конструкции с пояснительным заголовком
print(data['a']) # вывод только данных

