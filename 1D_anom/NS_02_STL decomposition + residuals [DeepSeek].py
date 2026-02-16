"""
Выделение аномалий 1D в нестационарных данных
2. STL‑декомпозиция + остатки
Суть: выделяем тренд и сезонность (STL), анализируем остатки.
Аномалии — большие отклонения остатков.

Эта программа выполняет следующие задачи:

Основные возможности:
1/ STL-декомпозиция - разложение временного ряда на тренд, сезонность и остатки
2/ Автоматическое определение периода сезонности, если он не указан
3/ Анализ остатков для выявления аномалий
4/ Статистические тесты для проверки свойств остатков
4/ Визуализация всех компонент и результатов

Ключевые параметры:
- period - период сезонности
- seasonal - длина сезонного сглаживателя
- robust - использование робастной версии STL
- anomaly_threshold - порог аномалий (в стандартных отклонениях)

Выходные данные:
- Компоненты декомпозиции (тренд, сезонность, остатки)
- Индексы обнаруженных аномалий
- Статистика остатков
- Визуализация результатов

Программа особенно полезна для анализа временных рядов с ярко выраженной сезонностью, таких как данные продаж, трафик, метеорологические наблюдения и т.д.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def stl_anomaly_detection(data, period=None, seasonal=13, robust=True,
                          anomaly_threshold=3, plot_results=True):
    """
    Выполняет STL-декомпозицию временного ряда и выделяет аномалии на основе остатков.

    Parameters:
    -----------
    data : array-like или pd.Series
        Входной временной ряд
    period : int, optional
        Период сезонности. Если None, определяется автоматически
    seasonal : int, optional
        Длина сезонного сглаживателя (должна быть нечетной)
    robust : bool, optional
        Использовать робастную версию STL
    anomaly_threshold : float, optional
        Порог для определения аномалий (в стандартных отклонениях)
    plot_results : bool, optional
        Отображать графики результатов

    Returns:
    --------
    dict : Словарь с результатами декомпозиции и индексами аномалий
    """

    # Преобразование входных данных
    if isinstance(data, pd.Series):
        series = data.copy()
    else:
        series = pd.Series(data)

    # Автоматическое определение периода, если не указан
    if period is None:
        # Пытаемся определить период по автокорреляции
        from statsmodels.tsa.stattools import acf
        acf_values = acf(series, nlags=min(len(series) // 2, 100))

        # Ищем локальные максимумы (исключая лаг 0)
        potential_periods = []
        for i in range(2, len(acf_values) - 1):
            if acf_values[i] > acf_values[i - 1] and acf_values[i] > acf_values[i + 1]:
                if acf_values[i] > 0.3:  # Порог значимости
                    potential_periods.append(i)

        if potential_periods:
            period = potential_periods[0]
            print(f"Автоматически определен период: {period}")
        else:
            period = 7  # По умолчанию недельный период
            print(f"Период не определен, используется значение по умолчанию: {period}")

    # Выполнение STL-декомпозиции
    stl = STL(series, period=period, seasonal=seasonal, robust=robust)
    result = stl.fit()

    # Извлечение компонентов
    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid

    # Статистический анализ остатков
    resid_mean = np.mean(resid)
    resid_std = np.std(resid)

    # Выявление аномалий на основе остатков
    anomalies = np.abs(resid - resid_mean) > anomaly_threshold * resid_std
    anomaly_indices = np.where(anomalies)[0]

    # Дополнительные статистики для остатков
    resid_skewness = stats.skew(resid.dropna())
    resid_kurtosis = stats.kurtosis(resid.dropna())

    # Тест на нормальность остатков
    _, normality_pvalue = stats.normaltest(resid.dropna())

    # Сбор результатов
    results = {
        'original': series,
        'trend': trend,
        'seasonal': seasonal,
        'resid': resid,
        'anomalies': anomalies,
        'anomaly_indices': anomaly_indices,
        'period': period,
        'resid_stats': {
            'mean': resid_mean,
            'std': resid_std,
            'skewness': resid_skewness,
            'kurtosis': resid_kurtosis,
            'normality_pvalue': normality_pvalue
        }
    }

    # Визуализация результатов
    if plot_results:
        plot_stl_decomposition(results, anomaly_threshold)

    return results


def plot_stl_decomposition(results, anomaly_threshold=3):
    """
    Визуализация результатов STL-декомпозиции и аномалий
    """
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))
    fig.suptitle(f'STL-декомпозиция и анализ аномалий\n'
                 f'Порог аномалий: {anomaly_threshold}σ',
                 fontsize=14, fontweight='bold')

    # Получаем индекс для временной шкалы
    if hasattr(results['original'], 'index'):
        index = results['original'].index
    else:
        index = range(len(results['original']))

    # Исходные данные
    axes[0].plot(index, results['original'], color='blue', alpha=0.7, label='Исходные данные')
    axes[0].set_ylabel('Исходный ряд')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Тренд
    axes[1].plot(index, results['trend'], color='green', label='Тренд')
    axes[1].set_ylabel('Тренд')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Сезонная компонента
    axes[2].plot(index, results['seasonal'], color='orange', label='Сезонность')
    axes[2].set_ylabel('Сезонность')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # Остатки и аномалии (ИСПРАВЛЕНО)
    resid = results['resid']
    anomalies = results['anomalies']

    # Преобразуем anomalies в массив той же длины, что и resid
    # Убеждаемся, что оба массива имеют одинаковую длину
    if len(anomalies) != len(resid):
        # Если есть различия в длине, корректируем
        min_len = min(len(anomalies), len(resid))
        anomalies = anomalies[:min_len]
        resid_aligned = resid.iloc[:min_len] if hasattr(resid, 'iloc') else resid[:min_len]
        index_aligned = index[:min_len]
    else:
        resid_aligned = resid
        index_aligned = index

    # График остатков
    axes[3].plot(index_aligned, resid_aligned, color='purple', alpha=0.7, label='Остатки')

    # Выделение аномалий
    if np.any(anomalies):
        # Создаем маску для аномалий
        anomaly_mask = anomalies

        # Для fill_between нужно указать x и y границы
        y_min = resid_aligned.min()
        y_max = resid_aligned.max()

        # Закрашиваем области аномалий
        axes[3].fill_between(index_aligned, y_min, y_max,
                             where=anomaly_mask,
                             color='red', alpha=0.3,
                             label='Аномалии',
                             interpolate=True)

    # Линии порогов
    axes[3].axhline(y=results['resid_stats']['mean'],
                    color='black', linestyle='-', alpha=0.5, label='Среднее')
    axes[3].axhline(y=results['resid_stats']['mean'] + anomaly_threshold * results['resid_stats']['std'],
                    color='red', linestyle='--', alpha=0.7, label=f'+{anomaly_threshold}σ')
    axes[3].axhline(y=results['resid_stats']['mean'] - anomaly_threshold * results['resid_stats']['std'],
                    color='red', linestyle='--', alpha=0.7, label=f'-{anomaly_threshold}σ')

    axes[3].set_ylabel('Остатки')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    # Гистограмма остатков
    axes[4].hist(resid.dropna(), bins=30, density=True, alpha=0.7, color='purple', edgecolor='black')

    # Добавляем нормальное распределение для сравнения
    x = np.linspace(resid.min(), resid.max(), 100)
    y = stats.norm.pdf(x, results['resid_stats']['mean'], results['resid_stats']['std'])
    axes[4].plot(x, y, 'r-', linewidth=2, label='Нормальное распределение')

    axes[4].axvline(x=results['resid_stats']['mean'] + anomaly_threshold * results['resid_stats']['std'],
                    color='red', linestyle='--', alpha=0.7)
    axes[4].axvline(x=results['resid_stats']['mean'] - anomaly_threshold * results['resid_stats']['std'],
                    color='red', linestyle='--', alpha=0.7)
    axes[4].set_xlabel('Значение остатков')
    axes[4].set_ylabel('Плотность')
    axes[4].set_title(f'Распределение остатков\n'
                      f'Среднее: {results["resid_stats"]["mean"]:.3f}, '
                      f'Стд: {results["resid_stats"]["std"]:.3f}\n'
                      f'Асимметрия: {results["resid_stats"]["skewness"]:.3f}, '
                      f'Эксцесс: {results["resid_stats"]["kurtosis"]:.3f}')
    axes[4].legend(loc='upper right')
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ... остальная часть функции без изменений
    # Вывод статистики
    print("\n" + "=" * 60)
    print("СТАТИСТИКА АНАЛИЗА АНОМАЛИЙ")
    print("=" * 60)
    print(f"Всего точек данных: {len(results['original'])}")
    print(f"Обнаружено аномалий: {len(results['anomaly_indices'])} "
          f"({len(results['anomaly_indices']) / len(results['original']) * 100:.2f}%)")
    print(f"\nСтатистика остатков:")
    print(f"  Среднее: {results['resid_stats']['mean']:.4f}")
    print(f"  Стандартное отклонение: {results['resid_stats']['std']:.4f}")
    print(f"  Асимметрия: {results['resid_stats']['skewness']:.4f}")
    print(f"  Эксцесс: {results['resid_stats']['kurtosis']:.4f}")
    print(f"  p-value теста на нормальность: {results['resid_stats']['normality_pvalue']:.4f}")

    if results['resid_stats']['normality_pvalue'] > 0.05:
        print("  ✓ Остатки распределены нормально (p > 0.05)")
    else:
        print("  ✗ Остатки НЕ распределены нормально (p < 0.05)")


def generate_sample_data(n_points=365, period=7, trend_coeff=0.01,
                         seasonal_amplitude=1, noise_level=0.3,
                         anomaly_prob=0.03, anomaly_amplitude=5):
    """
    Генерирует пример временного ряда с аномалиями для тестирования
    """
    np.random.seed(42)
    t = np.arange(n_points)

    # Тренд
    trend = trend_coeff * t

    # Сезонность
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / period)

    # Шум
    noise = noise_level * np.random.randn(n_points)

    # Аномалии
    anomalies = np.random.random(n_points) < anomaly_prob
    anomaly_effect = anomalies * anomaly_amplitude * np.random.randn(n_points)

    # Итоговый ряд
    data = trend + seasonal + noise + anomaly_effect

    return pd.Series(data, index=pd.date_range('2023-01-01', periods=n_points, freq='D'))


# Пример использования
if __name__ == "__main__":
    print("Генерация тестовых данных...")
    # Генерируем пример данных
    test_data = generate_sample_data(n_points=365, period=7)

    # Применяем STL-декомпозицию для поиска аномалий
    print("\nЗапуск STL-декомпозиции и анализа аномалий...")
    results = stl_anomaly_detection(
        test_data,
        period=7,  # Недельный период
        seasonal=13,  # Длина сезонного сглаживателя
        robust=True,
        anomaly_threshold=2.5,  # Порог аномалий
        plot_results=True
    )

    # Выводим индексы обнаруженных аномалий
    print("\n" + "=" * 60)
    print("ИНДЕКСЫ ОБНАРУЖЕННЫХ АНОМАЛИЙ:")
    print("=" * 60)
    if len(results['anomaly_indices']) > 0:
        # Показываем первые 10 аномалий
        for i, idx in enumerate(results['anomaly_indices'][:10]):
            date = test_data.index[idx] if hasattr(test_data.index, 'date') else idx
            value = test_data.iloc[idx]
            resid_value = results['resid'].iloc[idx]
            print(f"{i + 1}. Индекс: {idx}, Дата: {date}, Значение: {value:.3f}, "
                  f"Остаток: {resid_value:.3f}")

        if len(results['anomaly_indices']) > 10:
            print(f"... и еще {len(results['anomaly_indices']) - 10} аномалий")
    else:
        print("Аномалий не обнаружено")