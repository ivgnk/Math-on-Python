"""
Выделение аномалий
7. Автоэнкодер (Autoencoder)
Суть: нейронная сеть сжимает данные и восстанавливает их; большие ошибки восстановления — аномалии.

"""
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler

# 1. Генерация данных с аномалиями
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000).reshape(-1, 1)  # 1000 точек нормального распределения
anomalies = np.array([5, -6, 7, 6.5, -5.5]).reshape(-1, 1)  # явные аномалии
data = np.vstack([normal_data, anomalies])  # объединяем

# 2. Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# 3. Построение автоэнкодера
input_dim = X_scaled.shape[1]
encoding_dim = 1  # размер скрытого слоя (сжатое представление)

# Входной слой
input_layer = Input(shape=(input_dim,))

# Энкодер: сжимаем до encoding_dim
encoder = Dense(encoding_dim, activation='relu')(input_layer)

# Декодер: восстанавливаем исходную размерность
decoder = Dense(input_dim, activation='linear')(encoder)

# Полная модель: вход → энкодер → декодер → выход
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Компилируем модель
autoencoder.compile(optimizer='adam', loss='mse')  # MSE — средняя квадратичная ошибка


# 4. Обучение на "нормальных" данных (без явных аномалий)
# Берем только первые 1000 точек (без добавленных аномалий)
X_train = X_scaled[:1000]
autoencoder.fit(
    X_train,
    X_train,  # цель — восстановить тот же вход
    epochs=50,
    batch_size=16,
    shuffle=True,
    verbose=0  # не выводить лог обучения
)

# 5. Реконструкция всех данных (включая аномалии)
reconstructed = autoencoder.predict(X_scaled)

# 6. Вычисление ошибки реконструкции (MSE для каждой точки)
mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

# 7. Определение порога для аномалий
# Используем 95‑й процентиль ошибок на "нормальных" данных
threshold = np.percentile(mse[:1000], 95)
print(f"Порог аномалии (95‑й процентиль): {threshold:.4f}")

# 8. Маркировка аномалий
anomalies_mask = mse > threshold

# 9. Визуализация
plt.figure(figsize=(12, 6))

# График 1: Исходные данные с маркировкой аномалий
plt.subplot(1, 2, 1)
plt.scatter(
    range(len(data)),
    data,
    c=np.where(anomalies_mask, 'red', 'blue'),
    s=20,
    label='Данные'
)
plt.axhline(y=np.mean(data), color='gray', linestyle='--', label='Среднее')
plt.title('Исходные данные (красное — аномалии)')
plt.xlabel('Индекс точки')
plt.ylabel('Значение')
plt.legend()

# График 2: Ошибка реконструкции
plt.subplot(1, 2, 2)
plt.plot(mse, label='Ошибка реконструкции (MSE)', color='purple')
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Порог (95‑й перц.) = {threshold:.4f}')
plt.title('Ошибка реконструкции автоэнкодера')
plt.xlabel('Индекс точки')
plt.ylabel('MSE')
plt.legend()

plt.tight_layout()
plt.show()

# 10. Вывод найденных аномалий
print("\nНайденные аномалии (индекс, значение, ошибка MSE):")
for i in range(len(data)):
    if anomalies_mask[i]:
        print(f"Индекс {i}: значение = {data[i][0]:.2f}, MSE = {mse[i]:.4f}")