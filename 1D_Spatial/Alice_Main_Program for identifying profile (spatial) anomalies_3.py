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
from scipy.signal import savgol_filter
from scipy.ndimage import label

def generate_test_data(n_points=500, seed=42):
    """Генерирует тестовые гравиметрические данные: тренд + синусоида + шум."""
    np.random.seed(seed)
    x = np.linspace(0, 100, n_points)

    # Полиномиальный тренд (кубический)
    trend = 0.001 * x**3 - 0.2 * x**2 + 3 * x + 10
    # Синусоидальные аномалии
    sinusoid = 8 * np.sin(0.3 * x) + 5 * np.sin(0.1 * x + 1)
    # Случайный шум
    noise = 2 * np.random.randn(n_points)
    # Итоговый сигнал
    data = trend + sinusoid + noise

    return x, data, trend, sinusoid, noise

def calculate_convergence_criterion(zones_prev, zones_current):
    """Критерий сходимости: относительное изменение числа зон."""
    if len(zones_prev) == 0:
        return 1.0
    change = abs(len(zones_current) - len(zones_prev)) / len(zones_prev)
    return change

def detect_anomaly_zones_iterative(data, x,
                                  smooth_window=11, polyorder=2,
                  initial_threshold=2.5,
                  min_threshold=0.5,
                  max_threshold=4.0,
                  min_zone_length=5,
                  target_min_zones=20,
                  target_max_zones=40,
                  max_iterations=230):
    """
    Интерактивный алгоритм поиска разбиения на 20–40 аномальных зон с гарантированным результатом.
    """

    threshold = initial_threshold
    step = (initial_threshold - min_threshold) / max_iterations
    zones_history = []
    convergence_history = []

    # Окно для отображения информации
    fig_info, ax_info = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(bottom=0.2)
    text_box = ax_info.text(0.1, 0.5, '', transform=ax_info.transAxes, fontsize=12)
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis('off')

    best_zones = []
    best_threshold = initial_threshold
    best_diff = float('inf')

    for iteration in range(max_iterations):
        # Сглаживание
        smoothed = savgol_filter(data, window_length=smooth_window, polyorder=polyorder)

        # Вычисление Z‑score в скользящем окне
        window = 15
        z_scores = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            local_data = smoothed[start:end]  # используем сглаженные данные
            mu, sigma = np.mean(local_data), np.std(local_data)
            if sigma > 0:
                z_scores[i] = (smoothed[i] - mu) / sigma
            else:
                z_scores[i] = 0

        # Бинаризация аномальных точек
        anomaly_mask = np.abs(z_scores) > threshold

        # Группировка в зоны
        labeled, num_features = label(anomaly_mask)
        zones = []
        for i in range(1, num_features + 1):
            indices = np.where(labeled == i)[0]
            if len(indices) >= min_zone_length:
                start, end = indices[0], indices[-1]
                mean_anomaly = np.mean(data[start:end+1])
                zones.append((start, end, mean_anomaly))

        # Сортировка по силе аномалии (абсолютное значение)
        zones.sort(key=lambda z: abs(z[2]), reverse=True)

        # Сохраняем историю
        zones_history.append(zones)

        # Расчёт критерия сходимости
        if iteration > 0:
            convergence = calculate_convergence_criterion(zones_history[-2], zones)
        else:
            convergence = 1.0
        convergence_history.append(convergence)

        # Оценка качества решения
        current_count = len(zones)
        target_center = (target_min_zones + target_max_zones) / 2
        diff_to_target = abs(current_count - target_center)

        # Обновляем лучшее решение, если текущее ближе к целевому диапазону
        if diff_to_target < best_diff:
            best_diff = diff_to_target
            best_zones = zones
            best_threshold = threshold

        # Обновление информации в отдельном окне
        remaining_iterations = max_iterations - iteration - 1
        in_target_range = target_min_zones <= current_count <= target_max_zones

        info_text = (
            f"Итерация: {iteration + 1}/{max_iterations}\n"
            f"Число аномальных зон: {current_count}\n"
            f"Порог Z‑score: {threshold:.2f}\n"
            f"Критерий сходимости: {convergence:.4f}\n"
            f"Осталось итераций: {remaining_iterations}\n"
            f"В целевом диапазоне (20–40): {'Да' if in_target_range else 'Нет'}\n"
            f"Лучшее решение: {len(best_zones)} зон"
        )
        text_box.set_text(info_text)
        fig_info.canvas.draw()
        plt.pause(0.3)

        # Проверка условий остановки
        if in_target_range and convergence < 0.1:  # сходимость достигнута
            print(f"Алгоритм сошёлся на итерации {iteration + 1}")
            break

        # Адаптивная регулировка шага
        if current_count < target_min_zones:
            # Если зон слишком мало, уменьшаем порог быстрее
            threshold = max(threshold - 2 * step, min_threshold)
        elif current_count > target_max_zones:
            # Если зон слишком много, уменьшаем шаг снижения порога
            threshold = max(threshold - step / 2, min_threshold)
        else:
            # В пределах диапазона — стандартное уменьшение
            threshold = max(threshold - step, min_threshold)

    plt.close(fig_info)  # закрываем окно информации

    return best_zones, best_threshold, zones_history, convergence_history

def plot_final_results(x, data, zones, smooth_window=11, polyorder=2):
    """Финальная визуализация всех результатов после завершения алгоритма."""
    smoothed = savgol_filter(data, window_length=smooth_window, polyorder=polyorder)

    plt.figure(figsize=(14, 8))

    # Исходные данные и сглаженные
    plt.subplot(2, 1, 1)
    plt.plot(x, data, 'b-', alpha=0.5, label='Исходные данные')
    plt.plot(x, smoothed, 'r-', linewidth=2, label='Сглаженные (Савицкий‑Голей)')
    plt.title('Сглаживание фильтром Савицкого‑Голея')
    plt.xlabel('Расстояние')
    plt.ylabel('Значение поля')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Аномальные зоны
    plt.subplot(2, 1, 2)
    plt.plot(x, data, 'b-', linewidth=2, alpha=0.7, label='Исходные данные')

    colors = plt.cm.Set1(np.linspace(0, 1, len(zones)))
    for i, (start, end, mean_val) in enumerate(zones):
        plt.axvline(x[start], color=colors[i], linestyle='--', alpha=0.8, linewidth=2)
        plt.axvline(x[end], color=colors[i], linestyle='--', alpha=0.8, linewidth=2)
        # Подпись номера зоны
        mid_x = x[start] + (x[end] - x[start]) / 2
        plt.text(mid_x, np.max(data) * 0.95, f'Зона {i + 1}',
                 ha='center', va='top', fontsize=9, color=colors[i], fontweight='bold')
        # Дополнительная информация о зоне
        plt.text(mid_x, np.min(data) * 1.05, f'{mean_val:.1f}',
                 ha='center', va='bottom', fontsize=8, color=colors[i])

    plt.title('Выделение аномальных зон: финальный результат')
    plt.xlabel('Расстояние (условные единицы)')
    plt.ylabel('Значение поля (условные единицы)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_convergence_history(convergence_history, zones_history, target_min=20, target_max=40):
    """Визуализация истории сходимости алгоритма."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # График критерия сходимости
    ax1.plot(range(1, len(convergence_history) + 1), convergence_history, 'ro-', markersize=6)
    ax1.set_xlabel('Номер итерации')
    ax1.set_ylabel('Критерий сходимости')
    ax1.set_title('История сходимости алгоритма')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.1, color='g', linestyle='--', label='Порог сходимости (0.1)')
    ax1.legend()

    # График числа зон по итерациям
    zone_counts = [len(zones) for zones in zones_history]
    ax2.plot(range(1, len(zone_counts) + 1), zone_counts, 'bo-', markersize=6)
    ax2.set_xlabel('Номер итерации')
    ax2.set_ylabel('Число аномальных зон')
    ax2.set_title('Число аномальных зон по итерациям')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=target_min, color='r', linestyle='--', label=f'Минимум ({target_min})')
    ax2.axhline(y=target_max, color='r', linestyle='--', label=f'Максимум ({target_max})')
    # Выделяем лучшую итерацию
    best_idx = np.argmin([abs(len(z) - (target_min + target_max) / 2) for z in zones_history])
    ax2.axvline(x=best_idx + 1, color='orange', linestyle='-', linewidth=2,
                label=f'Лучшее решение ({len(zones_history[best_idx])} зон)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Генерация тестовых данных
    x, data, trend, sinusoid, noise = generate_test_data(n_points=500, seed=42)

    print("Запуск интерактивного поиска аномальных зон...")
    print("Цель: найти 20–40 аномальных зон")

    # Интерактивный поиск оптимальных зон
    optimal_zones, final_threshold, zones_history, convergence_history = detect_anomaly_zones_iterative(
        data, x,
        smooth_window=11,
        polyorder=2,
        initial_threshold=2.5,
        min_threshold=0.05,
        max_threshold=4.0,
        min_zone_length=5,
        target_min_zones=20,
        target_max_zones=40,
        max_iterations=230
    )

    # Вывод результатов
    print("\n" + "=" * 60)
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    print(f"Оптимальный порог Z‑score: {final_threshold:.2f}")
    print(f"Найдено аномальных зон: {len(optimal_zones)}")
    print(f"Целевой диапазон: 20–40 зон")

    if len(optimal_zones) == 0:
        print("ВНИМАНИЕ: аномальные зоны не найдены. Попробуйте изменить параметры алгоритма.")
    else:
        print("\nПОДРОБНАЯ ИНФОРМАЦИЯ ПО ЗОНАМ:")
        for i, (start, end, mean_val) in enumerate(optimal_zones):
            print(f"Зона {i + 1}:")
            print(f"  Координаты: {x[start]:.1f} – {x[end]:.1f}")
            print(f"  Длина: {end - start + 1} точек")
            print(f"  Среднее значение: {mean_val:.2f}")
            print("-" * 40)

    # Финальная визуализация всех результатов
    plot_final_results(x, data, optimal_zones, smooth_window=11, polyorder=2)

    # Визуализация истории сходимости
    plot_convergence_history(convergence_history, zones_history, 20, 40)

if __name__ == '__main__':
    main()
