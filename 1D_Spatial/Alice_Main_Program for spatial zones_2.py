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
Чтобы весь профиль без остатка состоял из 20-40 аномальных зон,
т.е. между ними не должно быть промежутков,
они зоны должны граничить или с краем профиля или с соседними зонами
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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

def split_into_contiguous_zones(data, x, target_min_zones=20, target_max_zones=40, max_iterations=50):
    """
    Разбивает весь профиль на 20–40 смежных аномальных зон без промежутков.
    """

    best_zones = []
    best_score = float('inf')

    fig_info, ax_info = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(bottom=0.2)
    text_box = ax_info.text(0.1, 0.5, '', transform=ax_info.transAxes, fontsize=12)
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis('off')

    for iteration in range(max_iterations):
        # Определяем число зон для текущей итерации
        n_zones = target_min_zones + (iteration % (target_max_zones - target_min_zones + 1))

        # Равномерное начальное разбиение
        zone_length = len(data) // n_zones
        boundaries = [i * zone_length for i in range(n_zones)]
        boundaries.append(len(data))  # Гарантируем, что последняя зона доходит до конца

        # Оптимизация границ методом динамического программирования
        optimized_boundaries = optimize_zone_boundaries(data, boundaries, n_zones)

        # Создаём зоны: end теперь указывает на последнюю точку зоны (включительно)
        zones = []
        for i in range(n_zones):
            start = optimized_boundaries[i]
            end = optimized_boundaries[i + 1] - 1  # Корректировка: end — последняя точка зоны
            if start <= end:  # Проверяем валидность зоны
                mean_val = np.mean(data[start:end + 1])
                zones.append((start, end, mean_val))

        # Оцениваем качество разбиения
        score = calculate_partition_score(data, zones)

        # Сохраняем лучшее решение
        if score < best_score:
            best_score = score
            best_zones = zones

        # Обновление информации в отдельном окне
        remaining_iterations = max_iterations - iteration - 1
        info_text = (
            f"Итерация: {iteration + 1}/{max_iterations}\n"
            f"Число зон: {n_zones}\n"
            f"Качество разбиения: {score:.2f}\n"
            f"Осталось итераций: {remaining_iterations}\n"
            f"Лучшее качество: {best_score:.2f}"
        )
        text_box.set_text(info_text)
        fig_info.canvas.draw()
        plt.pause(0.3)

    plt.close(fig_info)
    return best_zones

def optimize_zone_boundaries(data, initial_boundaries, n_zones):
    """Оптимизирует границы зон методом динамического программирования."""
    n = len(data)

    # Матрица стоимости: cost[i][j] = дисперсия участка от i до j
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            segment = data[i:j + 1]
            cost[i][j] = np.var(segment)

    # Динамическое программирование
    dp = np.full((n_zones + 1, n + 1), float('inf'))
    parent = np.zeros((n_zones + 1, n + 1), dtype=int)

    dp[0][0] = 0

    for k in range(1, n_zones + 1):
        for j in range(k, n + 1):
            for i in range(k - 1, j):
                current_cost = dp[k - 1][i] + cost[i][j - 1]
                if current_cost < dp[k][j]:
                    dp[k][j] = current_cost
                    parent[k][j] = i

    # Восстановление оптимальных границ
    boundaries = []
    current = n
    for k in range(n_zones, 0, -1):
        boundaries.append(current)
        current = parent[k][current]
    boundaries.append(0)
    boundaries.reverse()

    return boundaries

def calculate_partition_score(data, zones):
    """Рассчитывает качество разбиения как сумму внутризонных дисперсий."""
    total_score = 0
    for start, end, _ in zones:
        segment = data[start:end + 1]  # Включаем последнюю точку
        total_score += np.var(segment) if len(segment) > 1 else 0
    return total_score

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

    # Аномальные зоны (смежные)
    plt.subplot(2, 1, 2)
    plt.plot(x, data, 'b-', linewidth=2, alpha=0.7, label='Исходные данные')

    colors = plt.cm.Set1(np.linspace(0, 1, len(zones)))
    for i, (start, end, mean_val) in enumerate(zones):
        # Заливка зоны
        plt.axvspan(x[start], x[end], alpha=0.3, color=colors[i])
        # Границы зон
        plt.axvline(x[start], color=colors[i], linestyle='-', alpha=0.8, linewidth=2)
        if i == len(zones) - 1:  # последняя граница
            plt.axvline(x[end], color=colors[i], linestyle='-', alpha=0.8, linewidth=2)

        # Подпись номера зоны (только цифра, шрифт 7 пт)
        mid_x = x[start] + (x[end] - x[start]) / 2
        plt.text(mid_x, np.max(data)*0.95, f'{i+1}',
                 ha='center', va='top', fontsize=7, color=colors[i], fontweight='bold')

        # Вертикальное отображение среднего значения под зоной
        plt.text(
            mid_x,                          # координата X — центр зоны
            np.min(data) * 0.98,     # координата Y — чуть выше минимума данных
            f'{mean_val:.1f}',         # текст: среднее значение с 1 знаком после запятой
            ha='center',               # выравнивание по горизонтали: по центру
            va='top',                 # выравнивание по вертикали: сверху
            fontsize=7,               # размер шрифта
            color=colors[i],          # цвет как у зоны
            rotation=90,              # поворот на 90° (снизу вверх)
            rotation_mode='anchor'     # корректное позиционирование при повороте
        )

    plt.title('Разбиение профиля на смежные аномальные зоны (20–40 зон)')
    plt.xlabel('Расстояние (условные единицы)')
    plt.ylabel('Значение поля (условные единицы)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_partition_statistics(zones, data, x):
    """Визуализация статистики по разбиению."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Распределение длин зон
    lengths = [zone[1] - zone[0] + 1 for zone in zones]  # +1, т.к. end включён
    ax1.hist(lengths, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Длина зоны (количество точек)')
    ax1.set_ylabel('Частота')
    ax1.set_title('Распределение длин аномальных зон')
    ax1.grid(True, alpha=0.3)

    # Средние значения по зонам
    means = [zone[2] for zone in zones]
    zone_numbers = range(1, len(zones) + 1)
    ax2.bar(zone_numbers, means, color=plt.cm.Set1(np.linspace(0, 1, len(zones))), alpha=0.7)
    ax2.set_xlabel('Номер зоны')
    ax2.set_ylabel('Среднее значение поля')
    ax2.set_title('Средние значения по аномальным зонам')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Вывод статистики в консоль
    print("\nСТАТИСТИКА РАЗБИЕНИЯ:")
    print(f"Общее число зон: {len(zones)}")
    print(f"Средняя длина зоны: {np.mean(lengths):.1f} точек")
    print(f"Минимальная длина: {np.min(lengths)} точек")
    print(f"Максимальная длина: {np.max(lengths)} точек")
    print(f"Стандартное отклонение длин: {np.std(lengths):.2f}")
    print(f"Среднее значение поля по всем зонам: {np.mean(means):.2f}")

def check_continuity(zones, total_length):
    """Проверяет, что разбиение непрерывно и покрывает весь профиль."""
    if not zones:
        return False

    # Первая зона должна начинаться с 0
    if zones[0][0] != 0:
        return False

    # Последняя зона должна заканчиваться в конце профиля (индекс total_length-1)
    if zones[-1][1] != total_length - 1:
        return False

    # Все зоны должны граничить друг с другом
    for i in range(1, len(zones)):
        if zones[i][0] != zones[i-1][1] + 1:  # +1: конец предыдущей + 1 = начало следующей
            return False

    return True

def save_partition_to_file(zones, n_zones, x, data, filename):
    """Сохраняет разбиение в текстовый файл с информацией о параметрах."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"РАЗБИЕНИЕ НА {n_zones} ЗОН\n")
        f.write("=" * 50 + "\n\n")

        # Информация о каждой зоне
        for i, (start, end, mean_val) in enumerate(zones):
            f.write(f"Зона {i+1}:\n")
            f.write(f"  Начало координаты: {x[start]:.3f}\n")
            f.write(f"  Конец координаты: {x[end]:.3f}\n")
            f.write(f"  Длина зоны (точек): {end - start + 1}\n")
            f.write(f"  Среднее значение: {mean_val:.4f}\n")
            f.write(f"  Диапазон индексов: [{start}, {end}]\n")
            f.write("- " * 30 + "\n")

        # Итоговая статистика
        lengths = [zone[1] - zone[0] + 1 for zone in zones]
        means = [zone[2] for zone in zones]

        f.write("\nСТАТИСТИКА РАЗБИЕНИЯ:\n")
        f.write(f"Общее число зон: {len(zones)}\n")
        f.write(f"Средняя длина зоны: {np.mean(lengths):.2f} точек\n")
        f.write(f"Минимальная длина: {np.min(lengths)} точек\n")
        f.write(f"Максимальная длина: {np.max(lengths)} точек\n")
        f.write(f"Стандартное отклонение длин: {np.std(lengths):.2f}\n")
        f.write(f"Среднее значение поля по всем зонам: {np.mean(means):.4f}\n")
        f.write(f"Суммарная дисперсия разбиения: {calculate_partition_score(data, zones):.4f}\n\n")

        # Параметры генерации данных (для контекста)
        f.write("ПАРАМЕТРЫ ГЕНЕРАЦИИ ДАННЫХ:\n")
        f.write(f"Количество точек: {len(data)}\n")

def main():
    # Генерация тестовых данных
    x, data, trend, sinusoid, noise = generate_test_data(n_points=500, seed=42)

    print("Запуск алгоритма разбиения профиля на смежные зоны...")
    print("Цель: получить разбиения на 20–40 зон для сохранения")

    # Создаём папку для результатов, если её нет
    import os
    output_dir = "partition_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Результаты будут сохранены в папку: {output_dir}")

    # Перебираем все числа зон от 20 до 40
    for n_zones in range(20, 41):
        print(f"\nОбрабатывается разбиение на {n_zones} зон...")

        # Создаём начальные границы (равномерное разбиение)
        zone_length = len(data) // n_zones
        boundaries = [i * zone_length for i in range(n_zones)]
        boundaries.append(len(data))  # Гарантируем покрытие всего профиля

        # Оптимизируем границы
        optimized_boundaries = optimize_zone_boundaries(data, boundaries, n_zones)

        # Формируем зоны
        zones = []
        for i in range(n_zones):
            start = optimized_boundaries[i]
            end = optimized_boundaries[i + 1] - 1  # Последняя точка зоны
            if start <= end:  # Проверяем валидность зоны
                mean_val = np.mean(data[start:end + 1])
                zones.append((start, end, mean_val))

        # Проверяем непрерывность
        is_continuous = check_continuity(zones, len(data))
        if not is_continuous:
            print(f"ВНИМАНИЕ: Разбиение на {n_zones} зон не является непрерывным!")
            continue

        # Сохраняем в файл
        filename = os.path.join(output_dir, f"{n_zones}.txt")
        save_partition_to_file(zones, n_zones, x, data, filename)
        print(f"✓ Разбиение на {n_zones} зон сохранено в {filename}")

    print("\n" + "=" * 60)
    print("ВСЕ РАЗБИЕНИЯ УСПЕШНО СОХРАНЕНЫ!")
    print("=" * 60)

    # Финальная визуализация для лучшего разбиения (например, на 30 зон)
    print("\nГенерация финальной визуализации для разбиения на 30 зон...")
    zone_length_30 = len(data) // 30
    boundaries_30 = [i * zone_length_30 for i in range(30)]
    boundaries_30.append(len(data))
    optimized_30 = optimize_zone_boundaries(data, boundaries_30, 30)

    zones_30 = []
    for i in range(30):
        start = optimized_30[i]
        end = optimized_30[i + 1] - 1
        if start <= end:
            mean_val = np.mean(data[start:end + 1])
            zones_30.append((start, end, mean_val))

    plot_final_results(x, data, zones_30, smooth_window=11, polyorder=2)
    plot_partition_statistics(zones_30, data, x)

if __name__ == '__main__':
    main()
