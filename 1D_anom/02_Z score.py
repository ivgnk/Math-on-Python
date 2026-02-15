"""
Выделение аномалий
2. Z‑score (стандартные отклонения)
Суть: аномалии — точки, у которых Z‑score > 3 (или < −3),
т.е. они отстоят от среднего более чем на 3σ.
"""
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

# Генерируем данные с аномалиями
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
# добавляем явные аномалии
data = np.append(data, [5, -6, 7])

# Вычисляем Z‑score
z_scores = np.abs(stats.zscore(data))

# Пороговое значение
threshold = 3
anomalies_z = z_scores > threshold

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(range(len(data)), data, c=np.where(anomalies_z, 'red', 'blue'), s=20)
plt.axhline(np.mean(data) + 3 * np.std(data), color='red', linestyle='--', label='+3σ')
plt.axhline(np.mean(data) - 3 * np.std(data), color='red', linestyle='--', label='−3σ')
plt.title('Выделение аномалий по Z‑score')
plt.xlabel('Индекс точки'); plt.ylabel('Значение')
plt.grid(True); plt.legend(); plt.show()

