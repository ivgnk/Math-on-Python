"""
Выделение аномалий
5. Local Outlier Factor (LOF)
Суть: сравнивает локальную плотность точки с плотностью её соседей; низкие значения — аномалии.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Генерируем данные с аномалиями
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
# добавляем явные аномалии
data = np.append(data, [5, -6, 7])
X = data.reshape(-1, 1)

# Обучаем LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
anomaly_labels_lof = lof.fit_predict(X)  # -1 = аномалия, 1 = норма

anomalies_lof = anomaly_labels_lof == -1

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(range(len(data)), data, c=np.where(anomalies_lof, 'red', 'blue'), s=20)
plt.title('Выделение аномалий LOF'); plt.xlabel('Индекс точки'); plt.ylabel('Значение')
plt.grid(True); plt.show()
