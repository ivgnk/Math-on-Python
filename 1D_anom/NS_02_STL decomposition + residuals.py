"""
Выделение аномалий 1D в нестационарных данных
2. STL‑декомпозиция + остатки
Суть: выделяем тренд и сезонность (STL), анализируем остатки.
Аномалии — большие отклонения остатков.
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')  # Убираем предупреждения для чистоты вывода

def stl_anomaly_detection(data, period=None, threshold=3, robust=True):
    """
    Выделяет аномалии с помощью STL‑декомпозиции.

    Параметры:
    - data: одномерный массив данных
    - period: период сезонности (если None, попытаемся определить автоматически)
    - threshold: порог Z‑score для аномалий
    - robust: использовать ли устойчивую версию STL

    Возвращает:
    - anomalies: логический массив аномалий
    - trend, seasonal, resid: компоненты декомпозиции
    """
    # Проверка входных данных
    if len(data) < 10:
        raise ValueError("Недостаточно данных: нужно минимум 10 точек")

    # Автоопределение периода, если не задан
    if period is None:
        # Для демонстрации зададим период как 51 (нечётное) при длине ряда ≥ 150
        if len(data) >= 150:
            period = 51
        else:
            # Если данных мало, используем упрощённый подход без сезонности
            period = 1  # отключаем сезонность

    # Корректировка периода: должен быть нечётным ≥ 3
    if period < 3:
        period = 3
    elif period % 2 == 0:  # если чётный
        period += 1  # делаем нечётным

    # Проверка достаточности данных: минимум 2 полных цикла
    min_required = 2 * period
    if len(data) < min_required:
        print(f"Предупреждение: недостаточно данных для периода {period}. "
              f"Требуется {min_required}, доступно {len(data)}.")
        # Уменьшаем период до максимально возможного нечётного значения
        max_possible = len(data) // 2
        if max_possible < 3:
            raise ValueError("Слишком мало данных для STL‑декомпозиции")
        period = max_possible if max_possible % 2 == 1 else max_possible - 1
        print(f"Используем период {period} вместо {period}")

    try:
        # STL‑декомпозиция
        stl = STL(data, seasonal=period, robust=robust, period=period)
        result = stl.fit()

        trend = result.trend
        seasonal = result.seasonal
        resid = result.resid

        # Поиск аномалий в остатках (Z‑score)
        z_scores = np.abs((resid - np.mean(resid)) / np.std(resid))
        anomalies = z_scores > threshold

        return anomalies, trend, seasonal, resid, period

    except Exception as e:
        print(f"Критическая ошибка при STL‑декомпозиции: {e}")
        # В случае неудачи возвращаем пустые результаты
        return np.zeros(len(data), dtype=bool), data, np.zeros_like(data), data, None

# --- ГЕНЕРАЦИЯ ИДЕАЛЬНО ПОДХОДЯЩИХ ДАННЫХ---
np.random.seed(42)
t = np.arange(300)  # Достаточно данных для периода 51 (300 > 2*51 = 102)

# Нестационарный ряд с явной сезонностью и аномалиями
data = (0.02 * t) + 3 * np.sin(2 * np.pi * t / 51) + np.random.normal(0, 0.3, 300)
data[50] = 8    # аномалия (резкий скачок)
data[250] = -6  # аномалия (глубокий провал)

# ---ПРИМЕНЕНИЕ МЕТОДА---
print("Запуск STL‑декомпозиции...")
anomalies, trend, seasonal, resid, used_period = stl_anomaly_detection(
    data,
    period=51,  # явно задаём нечётный период
    threshold=2.5,
    robust=True
)

print(f"Использованный период сезонности: {used_period}")
print(f"Обнаружено аномалий: {np.sum(anomalies)}")

# ---ВИЗУАЛИЗАЦИЯ---
plt.figure(figsize=(14, 12))

# Подграфик 1: Исходные данные и аномалии
plt.subplot(4, 1, 1)
plt.plot(t, data, label='Исходные данные', color='blue', alpha=0.7)
plt.scatter(t[anomalies], data[anomalies], color='red', s=60, label='Аномалии')
plt.title(f'1. Исходные данные (период сезонности: {used_period})')
plt.ylabel('Значение')
plt.legend()

# Подграфик 2: Тренд
plt.subplot(4, 1, 2)
plt.plot(t, trend, label='Тренд', color='orange')
plt.title('2. Выявленный тренд (STL)')
plt.ylabel('Тренд')
plt.legend()

# Подграфик 3: Сезонность
plt.subplot(4, 1, 3)
plt.plot(t, seasonal, label='Сезонность', color='green')
plt.title('3. Выявленная сезонность (STL)')
plt.ylabel('Сезонность')
plt.legend()

# Подграфик 4: Остатки и пороги
plt.subplot(4, 1, 4)
plt.plot(t, resid, label='Остатки (данные − тренд − сезонность)', color='purple', alpha=0.7)
mad = np.median(np.abs(resid - np.median(resid)))
threshold_value = 2.5 * mad
plt.axhline(y=threshold_value, color='red', linestyle='--', label=f'Порог (+{2.5}×MAD)')
plt.axhline(y=-threshold_value, color='red', linestyle='--')
plt.scatter(t[anomalies], resid[anomalies], color='red', s=40, label='Аномалии в остатках')
plt.title('4. Остатки STL и пороги аномалий (на основе MAD)')
plt.xlabel('Время')
plt.ylabel('Остатки')
plt.legend()

plt.tight_layout()
plt.show()


# ---ВЫВОД РЕЗУЛЬТАТОВ---
print("\nДЕТАЛЬНАЯ ИНФОРМАЦИЯ:")
print(f"Индексы аномалий: {np.where(anomalies)[0]}")
print(f"Значения аномалий: {data[anomalies]}")

z_scores_full = np.abs((resid - np.mean(resid)) / np.std(resid))
anomaly_z_scores = z_scores_full[anomalies]

if len(anomaly_z_scores) > 0:
    print("Z‑scores аномалий:")
    for i, z_score in enumerate(anomaly_z_scores):
        print(f"  Точка {np.where(anomalies)[0][i]}: {z_score:.2f}")
else:
    print("Аномалий не обнаружено.")