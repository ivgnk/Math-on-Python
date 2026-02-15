"""
Выделение аномалий
3. Межквартильный размах (IQR)
Суть: аномалии — значения вне интервала [Q1 − 1.5·IQR; Q3 + 1.5·IQR]
"""
import numpy as np
import matplotlib.pyplot as plt

# Генерируем данные с аномалиями
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
# добавляем явные аномалии
data = np.append(data, [5, -6, 7])

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

# Границы аномалий
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

anomalies_iqr = (data < lower_bound) | (data > upper_bound)

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(range(len(data)), data, c=np.where(anomalies_iqr, 'red', 'blue'), s=20)
plt.axhline(lower_bound, color='orange', linestyle='--', label='Нижняя граница IQR')
plt.axhline(upper_bound, color='orange', linestyle='-.', label='Верхняя граница IQR')
plt.title('Выделение аномалий по IQR')
plt.xlabel('Индекс точки'); plt.ylabel('Значение')
plt.grid(True); plt.legend(); plt.show()
