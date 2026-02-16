"""
Выделение аномалий
4. Isolation Forest для одномерных нестационарных данных
Суть метода
Isolation Forest («изолирующий лес») — алгоритм обнаружения аномалий, который:
- не строит модель «нормальных» данных, а изолирует выбросы;
- использует ансамбль случайных деревьев (iTrees);
- аномалии обнаруживаются по короткой длине пути в дереве:
чем быстрее точка изолируется, тем она аномальнее.

Для одномерных данных алгоритм работает корректно, хотя чаще применяется к многомерным.
Ключевое преимущество — не требует предположений о распределении и устойчив к нестационарности.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def detect_anomalies_isolation_forest(data, contamination=0.05, random_state=42):
    """
    Выделяет аномалии в одномерном ряду с помощью Isolation Forest.
    Параметры:
    - data: одномерный массив (1D)
    - contamination: доля ожидаемых аномалий (0 < contamination <= 0.5)
    - random_state: для воспроизводимости
    Возвращает:
    - anomalies: логический массив (True = аномалия)
    - scores: оценки аномальности (чем ниже, тем аномальнее)
    """
    # Преобразуем в 2D-массив (sklearn требует форму [n_samples, n_features])
    X = data.reshape(-1, 1)
    # Инициализация модели
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,  # число деревьев в лесу
        max_samples='auto',  # размер подвыборки для каждого дерева
        verbose=0
    )
    # Обучение и предсказание
    y_pred = model.fit_predict(X)  # -1 = аномалия, 1 = норма
    scores = model.decision_function(X)  # оценка аномальности (чем ниже, тем хуже)
    # Преобразуем в булевый массив
    anomalies = y_pred == -1
    return anomalies, scores

# --- Случайные данные ---
np.random.seed(42)
t = np.arange(1000)
# Нестационарный ряд: тренд + шум + аномалии
data = 0.01 * t + np.sin(0.03 * t) + np.random.normal(0, 0.5, 1000)
data[100] = 8  # аномалия (резкий скачок)
data[900] = -7  # аномалия (глубокий провал)

# Вызов функции
anomalies, scores = detect_anomalies_isolation_forest(
    data,
    contamination=0.02,  # ожидаем 2% аномалий
    random_state=42
)

# --- Визуализация ---
plt.figure(figsize=(14, 8))

# Подграфик 1: Исходные данные и аномалии
plt.subplot(2, 1, 1)
plt.plot(t, data, label='Исходные данные', color='blue', alpha=0.7)
plt.scatter(t[anomalies], data[anomalies], color='red', s=60, label='Аномалии (Isolation Forest)')
plt.title('1. Исходные данные и обнаруженные аномалии')
plt.xlabel('Время'); plt.ylabel('Значение')
plt.grid(); plt.legend()

# Подграфик 2: Оценки аномальности
plt.subplot(2, 1, 2)
plt.plot(t, scores, label='Оценка аномальности (decision_function)', color='purple', alpha=0.8)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Порог (0)')
plt.scatter(t[anomalies], scores[anomalies], color='red', s=40, label='Аномалии')
plt.title('2. Оценки аномальности: чем ниже значение, тем аномальнее точка')
plt.xlabel('Время'); plt.ylabel('Оценка аномальности')
plt.grid(); plt.legend()
plt.tight_layout(); plt.show()

# --- Вывод результатов ---
print("Индексы аномалий:", np.where(anomalies)[0])
print("Значения аномалий:", data[anomalies])
print("Оценки аномальности (scores):", scores[anomalies])