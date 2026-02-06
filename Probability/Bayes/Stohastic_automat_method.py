"""
Программа на Питон нахождения глобального минимума неунимодальной функции методом стохастических автоматов

Ниже — программа на Python для поиска глобального минимума
неунимодальной функции методом стохастических автоматов (стохастической аппроксимации).

==Суть метода==
Метод стохастических автоматов относится к классу вероятностных алгоритмов глобальной оптимизации.
Он:
- не требует градиентов;
- устойчив к «шумным» функциям и множеству локальных экстремумов;
- итеративно улучшает решение, используя случайные возмущения и правила принятия/отклонения пробных точек.

==Алгоритм (упрощённая версия)==
1. Задать начальную точку x0, шаг h, вероятность перехода p.
2. На каждой итерации:
- сгенерировать случайное направление (возмущение);
- вычислить значение функции в новой точке;
- если значение улучшилось — принять новую точку;
- иначе — принять с вероятностью p (чтобы избежать застревания в локальном минимуме).

3. Постепенно уменьшать шаг h (охлаждение).
4. Повторять до сходимости.
"""

import numpy as np
import matplotlib.pyplot as plt

def stochastic_automaton_minimize(func, bounds, x0=None, n_iter=1000, h_init=1.0, p_accept=0.3, temp_decay=0.99):
    """
    Поиск глобального минимума методом стохастического автомата.

    Параметры:
    - func: целевая функция (скаляр → скаляр)
    - bounds: кортеж (low, high) — границы для каждой координаты
    - x0: начальная точка (если None — случайная внутри bounds)
    - n_iter: число итераций
    - h_init: начальный шаг возмущения
    - p_accept: вероятность принять ухудшение
    - temp_decay: коэффициент уменьшения шага (охлаждение)

    Возвращает:
    - x_best: найденный минимум
    - f_best: значение функции в минимуме
    - history: история точек и значений
    """
    low, high = np.array(bounds[0]), np.array(bounds[1])
    dim = len(low)

    # Начальная точка
    if x0 is None:
        x = low + np.random.rand(dim) * (high - low)
    else:
        x = np.array(x0)

    x_best = x.copy()
    f_best = func(x)

    h = h_init
    history = []

    for i in range(n_iter):
        # Случайное возмущение в пределах [-h, h]
        dx = np.random.uniform(-h, h, size=dim)
        x_new = x + dx

        # Проекция на допустимую область
        x_new = np.clip(x_new, low, high)

        f_new = func(x_new)

        # Принимаем, если лучше, или с вероятностью p_accept
        if f_new < f_best or np.random.rand() < p_accept:
            x = x_new
            if f_new < f_best:
                x_best = x_new.copy()
                f_best = f_new

        # Уменьшаем шаг (охлаждение)
        h *= temp_decay

        history.append((x.copy(), f_best))

    return x_best, f_best, history


# --- Пример использования ---

# Неунимодальная функция (много локальных минимумов)
def test_function(x):
    return np.sin(x[0]) * np.cos(x[1]) + 0.1 * (x[0] ** 2 + x[1] ** 2)


# Границы поиска
bounds = ([-5, -5], [5, 5])

# Запуск оптимизации
x_min, f_min, history = stochastic_automaton_minimize(
    func=test_function,
    bounds=bounds,
    x0=[0.0, 0.0],
    n_iter=2000,
    h_init=2.0,
    p_accept=0.2,
    temp_decay=0.98
)

print(f"Найденный минимум: x = {x_min}, f(x) = {f_min:.6f}")

# Визуализация истории поиска (для 2D)
if len(bounds[0]) == 2:
    xs, fs = zip(*history)
    xs = np.array(xs)
    plt.figure(figsize=(10, 6))
    plt.plot(fs, label="Текущее лучшее значение f(x)")
    plt.xlabel("Итерация")
    plt.ylabel("f(x)")
    plt.title("Сходимость стохастического автомата")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    Z = np.sin(X) * np.cos(Y) + 0.1 * (X ** 2 + Y ** 2)
    plt.contour(X, Y, Z, levels=20, alpha=0.6, colors='gray')
    plt.scatter(xs[:, 0], xs[:, 1], c=range(len(xs)), cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(label="Номер итерации")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Траектория поиска в пространстве параметров")
    plt.show()
