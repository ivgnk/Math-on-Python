"""
Alice
Выделение пространственных гравитационных аномалий: методика и алгоритмы
Суть задачи
Требуется выделить участки аномальных значений на профиле гравиметрических измерений, где:
- значения систематически выше/ниже соседних;
- аномальные зоны протяжённые (не точечные выбросы);
- общее число зон — не более 20–40.

Ключевое отличие от поиска отдельных пиков: ищем пространственно согласованные
аномальные сегменты, а не единичные экстремумы.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import label

def generate_test_data(n_points=500, seed=42):
    """Генерирует тестовые гравиметрические данные: тренд + синусоида + шум."""
    np.random.seed(seed)
    x = np.linspace(0, 100, n_points)

    # Полиномиальный тренд (например, кубический)
    trend = 0.001 * x ** 3 - 0.2 * x ** 2 + 3 * x + 10
    # Синусоидальные аномалии
    sinusoid = 8 * np.sin(0.3 * x) + 5 * np.sin(0.1 * x + 1)
    # Случайный шум
    noise = 2 * np.random.randn(n_points)
    # Итоговый сигнал
    data = trend + sinusoid + noise
    return x, data


def detect_anomaly_zones(data, x,
                         smooth_window=11, polyorder=2,
                         z_threshold=2.0,
                         min_zone_length=5,
                         max_zones=40):
    """
    Выделяет аномальные зоны на профиле.

    Параметры:
    - data: массив значений (гравиметрия)
    - x: координаты точек
    - smooth_window: окно сглаживания (нечётное)
    - polyorder: порядок полинома для фильтра Савицкого‑Голея
    - z_threshold: порог Z‑score для аномалии
    - min_zone_length: мин. длина зоны (точек)
    - max_zones: макс. число зон для возврата

    Возвращает: список (start_idx, end_idx, mean_anomaly)
    """
    # 1. Сглаживание
    smoothed = savgol_filter(data, window_length=smooth_window, polyorder=polyorder)

    # 2. Вычисление отклонений (Z‑score в скользящем окне)
    window = 15
    z_scores = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        local_data = data[start:end]
        mu, sigma = np.mean(local_data), np.std(local_data)
        if sigma > 0:
            z_scores[i] = (data[i] - mu) / sigma
        else:
            z_scores[i] = 0

    # 3. Бинаризация: аномальные точки
    anomaly_mask = np.abs(z_scores) > z_threshold

    # 4. Группировка в зоны
    labeled, num_features = label(anomaly_mask)
    zones = []
    for i in range(1, num_features + 1):
        indices = np.where(labeled == i)[0]
        if len(indices) >= min_zone_length:
            start, end = indices[0], indices[-1]
            mean_anomaly = np.mean(data[start:end + 1])
            zones.append((start, end, mean_anomaly))

    # 5. Сортировка по силе аномалии и ограничение числа
    zones.sort(key=lambda z: abs(z[2]), reverse=True)  # по |среднего значения|
    zones = zones[:max_zones]

    return zones


def plot_results(x, data, zones):
    """Визуализирует исходные данные и границы аномальных зон."""
    plt.figure(figsize=(14, 6))
    plt.plot(x, data, label='Исходные данные', color='blue', alpha=0.7)

    # Вертикальные линии на границах зон
    for start, end, _ in zones:
        plt.axvline(x[start], color='red', linestyle='--', alpha=0.6)
        plt.axvline(x[end], color='red', linestyle='--', alpha=0.6)
        # Подпись номера зоны
        mid_x = x[start] + (x[end] - x[start]) / 2
        plt.text(mid_x, np.max(data) * 0.95, f'Зона {len(zones)}',
                 ha='center', va='top', fontsize=9, color='red')

    plt.xlabel('Расстояние (условные единицы)')
    plt.ylabel('Значение поля (условные единицы)')
    plt.title('Выделение аномальных зон')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    # Генерация тестовых данных
    x, data = generate_test_data(n_points=500, seed=42)

    # Выделение аномальных зон
    zones = detect_anomaly_zones(
        data, x,
        smooth_window=11,
        polyorder=2,
        z_threshold=2.0,
        min_zone_length=5,
        max_zones=30
    )

    # Вывод результатов
    print("Найденные аномальные зоны:")
    for i, (start, end, mean_val) in enumerate(zones):
        print(f"Зона {i + 1}: {x[start]:.1f} – {x[end]:.1f} "
              f"(длина: {end - start + 1} т., среднее: {mean_val:.2f})")

    # Визуализация
    plot_results(x, data, zones)


if __name__ == '__main__':
    main()
