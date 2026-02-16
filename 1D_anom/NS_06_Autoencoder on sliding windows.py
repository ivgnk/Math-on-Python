"""
Выделение аномалий 1D в нестационарных данных
6. Автоэнкодер с окном (Autoencoder on sliding windows) для нестационарных данных
Суть метода
Автоэнкодер — нейронная сеть, которая:
- Сжимает входные данные в низкоразмерное «скрытое представление» (latent space).
- Восстанавливает исходные данные из сжатого кода.
- Выявляет аномалии по ошибке восстановления: если сеть плохо восстанавливает точку, она, вероятно, аномальна.

Для временных рядов применяем скользящее окно:
разбиваем ряд на фрагменты фиксированной длины и подаём их на вход автоэнкодеру.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


def create_sliding_windows(data, window_size):
    """
    Создаёт скользящие окна из временного ряда.

    Параметры:
    - data: одномерный массив
    - window_size: длина окна (целое число)

    Возвращает:
    - X: массив форм (n_windows, window_size)
    """
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:(i + window_size)])
    return np.array(X)


def build_autoencoder(input_dim, encoding_dim=8):
    """
    Строит автоэнкодер: вход → кодировщик → декодировщик → выход.

    Параметры:
    - input_dim: размер входного окна
    - encoding_dim: размер скрытого слоя (меньше input_dim)

    Возвращает:
    - autoencoder: полная модель
    - encoder: только кодировщик
    """
    # Входной слой
    input_layer = Input(shape=(input_dim,))

    # Кодировщик (сжатие)
    encoded = Dense(encoding_dim, activation='relu')(input_layer)

    # Декодировщик (восстановление)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # Полная модель
    autoencoder = Model(inputs=input_layer, outputs=decoded)

    # Модель кодировщика
    encoder = Model(inputs=input_layer, outputs=encoded)

    return autoencoder, encoder


def detect_anomalies_autoencoder(data, window_size=50, encoding_dim=8,
                                 threshold_factor=3, epochs=50, batch_size=32):
    """
    Выявляет аномалии с помощью автоэнкодера на скользящих окнах.

    Параметры:
    - data: одномерный временной ряд
    - window_size: длина окна
    - encoding_dim: размер скрытого пространства
    - threshold_factor: множитель для порога (обычно 2–3)
    - epochs: число эпох обучения
    - batch_size: размер батча

    Возвращает:
    - anomalies: логический массив аномалий
    - reconstruction_error: ошибка восстановления для каждой точки
    - autoencoder: обученная модель
    """
    # 1. Создание скользящих окон
    X = create_sliding_windows(data, window_size)

    # 2. Нормализация данных
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Построение и обучение автоэнкодера
    autoencoder, encoder = build_autoencoder(window_size, encoding_dim)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    print("Обучение автоэнкодера...")
    autoencoder.fit(X_scaled, X_scaled,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=0)

    # 4. Восстановление данных и расчёт ошибки
    X_reconstructed = autoencoder.predict(X_scaled)
    reconstruction_error = np.mean(np.square(X_scaled - X_reconstructed), axis=1)

    # 5. Определение порога и аномалий
    mad = np.median(np.abs(reconstruction_error - np.median(reconstruction_error)))
    threshold = threshold_factor * mad
    anomalies_window = reconstruction_error > threshold

    # 6. Преобразование аномалий окон в аномалии точек
    # Каждая точка входит в несколько окон; считаем её аномальной, если хотя бы в одном окне она аномальна
    anomalies = np.zeros(len(data), dtype=bool)
    for i, is_anomaly in enumerate(anomalies_window):
        if is_anomaly:
            # Маркируем все точки в этом окне как потенциально аномальные
            start = i
            end = i + window_size
            anomalies[start:end] = True

    return anomalies, reconstruction_error, autoencoder


# --- Пример использования ---
np.random.seed(42)
t = np.arange(500)

# Нестационарный ряд: тренд + шум + аномалии
data = 0.015 * t + np.sin(0.04 * t) + np.random.normal(0, 0.3, 500)
data[100] = 6  # аномалия (резкий скачок)
data[400] = -5  # аномалия (глубокий провал)

# Вызов функции
anomalies, reconstruction_error, model = detect_anomalies_autoencoder(
    data,
    window_size=40,
    encoding_dim=10,
    threshold_factor=2.5,
    epochs=100,
    batch_size=16
)

# --- Визуализация ---
plt.figure(figsize=(14, 10))

# Подграфик 1: Исходные данные и аномалии
plt.subplot(3, 1, 1)
plt.plot(t, data, label='Исходные данные', color='blue', alpha=0.7)
plt.scatter(t[anomalies], data[anomalies], color='red', s=60, label='Аномалии')
plt.title('1. Исходные данные и обнаруженные аномалии')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()

# Подграфик 2: Ошибка восстановления
plt.subplot(3, 1, 2)
window_positions = np.arange(len(reconstruction_error)) + 20  # центр окна
plt.plot(window_positions, reconstruction_error, label='Ошибка восстановления', color='purple', alpha=0.8)
mad = np.median(np.abs(reconstruction_error - np.median(reconstruction_error)))
threshold = 2.5 * mad
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Порог ({2.5}×MAD)')
plt.title('2. Ошибка восстановления по окнам')
plt.xlabel('Позиция окна (центр)')
plt.ylabel('MSE ошибки')
plt.legend()

# Подграфик 3: Скрытое представление (пример для первых 100 окон)
plt.subplot(3, 1, 3)
X_windows = create_sliding_windows(data, 40)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_windows[:100])  # первые 100 окон
encoder = Model(inputs=model.input, outputs=model.layers[1].output)
latent_repr = encoder.predict(X_scaled)
for i in range(min(5, latent_repr.shape[1])):  # первые 5 компонент
    plt.plot(latent_repr[:, i], label=f'Компонента {i + 1}')
plt.title('3. Пример скрытого представления (первые 5 компонент)')
plt.xlabel('Номер окна')
plt.ylabel('Значение в скрытом пространстве')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# --- Вывод результатов ---
print("ИНФОРМАЦИЯ О РЕЗУЛЬТАТАХ:")
print(f"Общее число точек: {len(data)}")
print(f"Число аномалий: {np.sum(anomalies)}")