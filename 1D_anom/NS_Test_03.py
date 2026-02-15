"""
Проверка на нестационарность ряда
3. Тест KPSS (Kwiatkowski‑Phillips‑Schmidt‑Shin)
Суть: обратный ADF. Проверяет стационарность вокруг детерминированного тренда.
H₀: ряд стационарен (или стационарен вокруг тренда).
H₁: ряд нестационарен.
Если p‑value < 0,05, отвергаем H₀ → ряд нестационарен.

Интерпретация:
- Для случайного блуждания: p‑значение < 0,05 → отвергаем H₀ → ряд нестационарен.
- Если ADF говорит «нестационарен», а KPSS — «стационарен», это конфликт; тогда смотрят на ACF и контекст данных.
"""
import numpy as np
from statsmodels.tsa.stattools import kpss

# Пример данных (случайное блуждание — нестационарный ряд)
np.random.seed(42)
n = 1000
data = np.cumsum(np.random.randn(n))  # Интегрированный шум → нестационарность

def kpss_test(series, alpha=0.05, regression='c'):
    # regression='c' — проверяем стационарность вокруг константы
    # regression='ct' — вокруг линейного тренда
    result = kpss(series, regression=regression, nlags='auto')
    print('Результаты теста KPSS:')
    print(f'Статистика теста: {result[0]:.4f}')
    print(f'p‑значение: {result[1]:.4f}')
    print(f'Использовано задержек: {result[2]}')
    print(f'Тип регрессии: {regression}')

    if result[1] <= alpha:
        print('→ Ряд нестационарен (H₀ отвергнута)')
    else:
        print('→ Ряд стационарен (H₀ не отвергнута)')

# Применяем
kpss_test(data, alpha=0.05, regression='c')
