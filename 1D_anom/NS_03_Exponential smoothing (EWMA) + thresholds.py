"""
Выделение аномалий
3. Экспоненциальное сглаживание (EWMA) + пороги
Суть: моделируем локальное среднее через EWMA, аномалии — большие отклонения от прогноза.
"""
import numpy as np
import matplotlib.pyplot as plt

def ewma_anomalies(data, alpha=0.1, threshold_factor=3):
    """
    Выделяет аномалии с помощью EWMA (экспоненциально взвешенного скользящего среднего).
    Параметры:
    - data: массив значений (1D)
    - alpha: коэффициент сглаживания (0 < alpha <= 1)
    - threshold_factor: множитель для порога (обычно 2–3)
    Возвращает:
    - anomalies: логический массив (True = аномалия)
    - ewma: сглаженные значения
    - resid: остатки (data - ewma)
    """
    n = len(data)
    ewma = np.zeros(n)
    # Инициализация: первое значение = исходному
    ewma[0] = data[0]
    # Расчёт EWMA для всех точек
    for i in range(1, n):
        ewma[i] = alpha * data[i] + (1 - alpha) * ewma[i - 1]
    # Остатки: отклонение от сглаженной кривой
    resid = data - ewma
    # Адаптивный порог на основе MAD остатков
    mad = np.median(np.abs(resid - np.median(resid)))  # Median Absolute Deviation
    threshold = threshold_factor * mad
    # Маркировка аномалий
    anomalies = np.abs(resid) > threshold
    return anomalies, ewma, resid

# --- Пример использования ---
np.random.seed(42)
t = np.arange(1000)

# Нестационарный ряд: тренд + шум + аномалии
data = 0.015 * t + np.sin(0.04 * t) + np.random.normal(0, 0.4, 1000)
data[150] = 7  # аномалия (резкий скачок)
data[850] = -6  # аномалия (глубокий провал)

# Вызов функции
anomalies, ewma, resid = ewma_anomalies(data, alpha=0.2, threshold_factor=3)

# --- Визуализация ---
plt.figure(figsize=(14, 8))

# Подграфик 1: Исходные данные и аномалии
plt.subplot(2, 1, 1)
plt.plot(t, data, label='Исходные данные', color='blue', alpha=0.7)
plt.scatter(t[anomalies], data[anomalies], color='red', s=50, label='Аномалии')
plt.plot(t, ewma, label='EWMA (сглаживание)', color='orange', linewidth=2)
plt.title('1. Исходные данные и сглаженная кривая (EWMA)')
plt.xlabel('Время'); plt.ylabel('Значение')
plt.legend(); plt.grid(True)
plt.subplot(2, 1, 2)

# Подграфик 2: Остатки и пороги
plt.subplot(2, 1, 2)
plt.plot(t, resid, label='Остатки (data − EWMA)', color='purple', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axhline(y=np.median(resid) + 3 * np.median(np.abs(resid - np.median(resid))),
            color='red', linestyle='--', label='Верхний порог')
plt.axhline(y=np.median(resid) - 3 * np.median(np.abs(resid - np.median(resid))),
            color='red', linestyle='--', label='Нижний порог')
plt.scatter(t[anomalies], resid[anomalies], color='red', s=30, label='Аномалии в остатках')
plt.title('2. Остатки и адаптивные пороги (на основе MAD)')
plt.xlabel('Время'); plt.ylabel('Остаток')
plt.legend(); plt.grid(True)

plt.tight_layout(); plt.show()

# --- Вывод результатов ---
print("Индексы аномалий:", np.where(anomalies)[0])
print("Значения аномалий:", data[anomalies])
print("Отклонения (остатки):", resid[anomalies])