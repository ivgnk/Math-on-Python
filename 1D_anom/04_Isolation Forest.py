"""
Выделение аномалий
4. Isolation Forest
Суть: ансамбль «изолирующих деревьев» оценивает, насколько легко изолировать точку; низкие оценки — аномалии.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Генерируем данные с аномалиями
np.random.seed(42)
data = np.random.normal(0, 1, 1000)
# добавляем явные аномалии
data = np.append(data, [5, -6, 7])

# Преобразуем в столбец
X = data.reshape(-1, 1)

# Обучаем модель
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_labels = iso_forest.fit_predict(X)  # -1 = аномалия, 1 = норма


anomalies_if = anomaly_labels == -1

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(range(len(data)), data, c=np.where(anomalies_if, 'red', 'blue'), s=20)
plt.title('Выделение аномалий Isolation Forest')
plt.xlabel('Индекс точки'); plt.ylabel('Значение')
plt.grid(True); plt.show()