"""
Выделение аномалий 1D в нестационарных данных
2. STL‑декомпозиция + остатки
Суть: выделяем тренд и сезонность (STL), анализируем остатки.
Аномалии — большие отклонения остатков.
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL


# 1. Генерация данных (пример)
np.random.seed(42)
t = np.arange(1000)
data = (0.02 * t) + 2 * np.sin(0.06 * t) + np.random.normal(0, 0.3, 1000)
data[200] = 8   # аномалия
data[800] = -6  # аномалия


# 2. Параметры декомпозиции
period = 50  # задайте период сезонности


# Проверка: достаточно ли данных?
if len(data) < 2 * period:
    raise ValueError(f"Недостаточно данных: нужно минимум 2 полных цикла. "
                    f"Длина ряда: {len(data)}, период: {period} → требуется >= {2 * period}")


# 3. STL‑декомпозиция
try:
    stl = STL(data, seasonal=period, robust=True, period=period)  # явно указываем period
    result = stl.fit()
except Exception as e:
    print(f"Ошибка при STL‑декомпозиции: {e}")
    # Альтернатива: попробуем автоопределение периода
    try:
        stl = STL(data, robust=True)  # без явного period
        result = stl.fit()
        print("STL: период автоматически определён.")
    except Exception as e2:
        raise RuntimeError(f"Не удалось выполнить STL: {e2}")


# Теперь доступны компоненты
trend = result.trend
seasonal = result.seasonal
resid = result.resid

# 4. Поиск аномалий в остатках (Z‑score)
z_scores = np.abs((resid - np.mean(resid)) / np.std(resid))
threshold = 3
anomalies = z_scores > threshold


# 5. Визуализация (как в предыдущем примере)
plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(t, data, label='Исходные данные', color='blue', alpha=0.7)
plt.scatter(t[anomalies], data[anomalies], color='red', s=50, label='Аномалии')
plt.title('1. Исходные данные с аномалиями')
plt.ylabel('Значение')
plt.legend()


plt.subplot(4, 1, 2)
plt.plot(t, trend, label='Тренд', color='orange')
plt.title('2. Тренд (STL)')
plt.ylabel('Тренд')
plt.legend()


plt.subplot(4, 1, 3)
plt.plot(t, seasonal, label='Сезонность', color='green')
plt.title('3. Сезонность (STL)')
plt.ylabel('Сезонность')
plt.legend()


plt.subplot(4, 1, 4)
plt.plot(t, resid, label='Остатки', color='purple', alpha=0.7)
plt.axhline(y=threshold * np.std(resid), color='red', linestyle='--', label=f'Порог (+{threshold}σ)')
plt.axhline(y=-threshold * np.std(resid), color='red', linestyle='--', label=f'Порог (−{threshold}σ)')
plt.scatter(t[anomalies], resid[anomalies], color='red', s=30, label='Аномалии в остатках')
plt.title('4. Остатки и пороги аномалий')
plt.xlabel('Время')
plt.ylabel('Остатки')
plt.legend()

plt.tight_layout()
plt.show()

# 6. Вывод результатов
print("Индексы аномалий:", np.where(anomalies)[0])
print("Значения аномалий:", data[anomalies])
print("Z‑scores аномалий:", z_scores[anomalies])
