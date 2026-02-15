"""
Выделение аномалий 1D в нестационарных данных
1. Скользящее окно + Z‑score (адаптивный)
Суть: для каждой точки считаем Z‑score относительно локального окна (а не всего ряда).
Учитывает дрейф среднего и дисперсии.
"""

import numpy as np
import matplotlib.pyplot as plt

def rolling_zscore(data, window=20, threshold=3):
    z_scores = np.zeros_like(data)
    for i in range(len(data)):
        # Берём окно: от max(0, i-window//2) до min(len, i+window//2)
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        window_data = data[start:end]

        mean = np.mean(window_data)
        std = np.std(window_data)
        if std > 0:
            z_scores[i] = (data[i] - mean) / std
        else:
            z_scores[i] = 0
    return np.abs(z_scores) > threshold

# Случайные данные
np.random.seed(42)
t = np.arange(1000)
# Нестационарный ряд: тренд + шум + аномалии
data = 0.01 * t + np.sin(0.05 * t) + np.random.normal(0, 0.5, 1000)
data[200] = 5  # аномалия
data[800] = -4  # аномалия

anomalies = rolling_zscore(data, window=50, threshold=2.5)

plt.figure(figsize=(12, 4))
plt.plot(t, data, label='Данные', color='blue', alpha=0.7)
plt.scatter(t[anomalies], data[anomalies], color='red', s=50, label='Аномалии')
plt.title('Скользящий Z‑score')
plt.legend(); plt.grid(True); plt.show()
