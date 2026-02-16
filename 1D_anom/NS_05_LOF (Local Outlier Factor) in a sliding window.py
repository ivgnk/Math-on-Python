"""
Выделение аномалий 1D в нестационарных данных
5. LOF (Local Outlier Factor) в скользящем окне
Суть: в каждом окне ищем точки с низкой локальной плотностью (LOF).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# Случайные данные
np.random.seed(42)
t = np.arange(1000)
# Нестационарный ряд: тренд + шум + аномалии
data = 0.01 * t + np.sin(0.05 * t) + np.random.normal(0, 0.5, 1000)
data[200] = 5  # аномалия
data[800] = -4  # аномалия

def lof_rolling(data, window=30, contamination=0.1):
    anomalies = np.zeros(len(data), dtype=bool)
    for i in range(0, len(data), window):
        start, end = i, min(i + window, len(data))
        if end - start < 2:
            continue
        X_window = data[start:end].reshape(-1, 1)
        lof = LocalOutlierFactor(n_neighbors=min(10, len(X_window)-1), contamination=contamination)
        labels = lof.fit_predict(X_window)
        anomalies[start:end] = labels == -1
    return anomalies

anomalies_lof = lof_rolling(data, window=40, contamination=0.05)
plt.figure(figsize=(12, 4))
plt.plot(data, label='Данные', color='blue', alpha=0.7)
plt.scatter(np.where(anomalies_lof)[0], data[anomalies_lof], color='red', s=50, label='Аномалии')
plt.title('LOF в скользящем окне')
plt.legend(); plt.grid(); plt.show()
