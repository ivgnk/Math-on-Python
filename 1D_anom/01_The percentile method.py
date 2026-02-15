"""
Выделение аномалий
1. Метод процентилей (Percentile-based)
Суть: аномалии — значения, выходящие за заданный процентильный интервал
(например, ниже 5‑го или выше 95‑го процентиля).
"""
import numpy as np
import matplotlib.pyplot as plt

# Генерируем данные с аномалиями
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
data = np.append(data, [5, -6, 7])  # добавляем явные аномалии

# Определяем границы по процентилям
lower_percentile = np.percentile(data, 5)
upper_percentile = np.percentile(data, 95)

# Находим аномалии
anomalies = (data < lower_percentile) | (data > upper_percentile)

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(range(len(data)), data, c=np.where(anomalies, 'red', 'blue'), s=20)
plt.axhline(lower_percentile, color='gray', linestyle='--', label='5-й процентиль')
plt.axhline(upper_percentile, color='gray', linestyle='--', label='95-й процентиль')
plt.title('Выделение аномалий по процентилям')
plt.xlabel('Индекс точки'); plt.ylabel('Значение')
plt.grid(True); plt.legend(); plt.show()

