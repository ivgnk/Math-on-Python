"""
Выделение аномалий
6. Gaussian Mixture Model (GMM)
Суть: моделируем данные как смесь гауссиан; точки с низкой вероятностью принадлежности — аномалии."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Генерируем данные с аномалиями
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
# добавляем явные аномалии
data = np.append(data, [5, -6, 7])
X = data.reshape(-1, 1)

# Обучаем GMM
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(X)

# Логическая вероятность (log-likelihood)
log_prob = gmm.score_samples(X)
threshold_gmm = np.percentile(log_prob, 5)  # 5-й процентиль как порог
anomalies_gmm = log_prob < threshold_gmm

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(range(len(data)), data, c=np.where(anomalies_gmm, 'red', 'blue'), s=20)
plt.title('Выделение аномалий GMM'); plt.xlabel('Индекс точки'); plt.ylabel('Значение')
plt.grid(True); plt.show()
