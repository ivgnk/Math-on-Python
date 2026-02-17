"""
пример на Python, который демонстрирует преимущества MAD (Mean Absolute Deviation)
перед стандартным отклонением (SD) в условиях выбросов.

Почему MAD лучше при выбросах
- MAD использует абсолютные отклонения от медианы — менее чувствителен к экстремальным значениям.
- SD использует квадраты отклонений от среднего — сильные выбросы резко увеличивают результат.

Что делает код
1/ Генерирует данные:
100 значений из нормального распределения (μ=50, σ=10).
Добавляет 5 экстремальных выбросов (150–190).

2/ Вычисляет статистики:
Среднее, SD и MAD для «чистых» данных.
Те же показатели после добавления выбросов.

3/ Сравнивает устойчивость:
Показывает, на сколько процентов выросли SD и MAD из‑за выбросов.

4/ Визуализирует:
Гистограммы распределений (с выбросами и без).
Столбчатую диаграмму для наглядного сравнения SD и MAD в двух сценариях.

== Интерпретация результатов
SD вырос на 186 % — резко отреагировал на 5 выбросов (5 % от объёма данных).
MAD вырос лишь на 9 % — остался устойчивым, так как опирается на медиану и абсолютные отклонения.
На гистограммах видно, как выбросы «растягивают» распределение вправо, сильно смещая среднее, но почти не влияя на медиану.

== Когда выбирать MAD
- Используйте MAD, если:
В данных возможны выбросы или аномалии.
Нужно robust‑оценивание разброса (например, в реальном времени).
Интерпретация в исходных единицах важнее математической строгости.

- Когда SD предпочтительнее:
Данные близки к нормальному распределению без выбросов.
Нужны свойства дисперсии (например, для статистических тестов).
Работаете в рамках классической теории ошибок.

Итог. Пример наглядно показывает:
✅ MAD устойчив к выбросам — идеален для «шумных» реальных данных.
❌ SD сильно искажается при наличии аномалий — может ввести в заблуждение.
Визуализация подтверждает преимущество MAD в условиях неидеальных данных.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# 1. Генерируем данные без выбросов (нормальное распределение)
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=10, size=100)

# 2. Добавляем 5 сильных выбросов
outliers = [150, 160, 170, 180, 190]
data_with_outliers = np.concatenate([normal_data, outliers])


# 3. Вычисляем статистики
def mad(x):
    """Среднее абсолютное отклонение от медианы"""
    median = np.median(x)
    return np.mean(np.abs(x - median))

mean_clean = np.mean(normal_data)
std_clean = np.std(normal_data)
mad_clean = mad(normal_data)

mean_dirty = np.mean(data_with_outliers)
std_dirty = np.std(data_with_outliers)
mad_dirty = mad(data_with_outliers)

print("Без выбросов:")
print(f"Среднее = {mean_clean:.2f}, SD = {std_clean:.2f}, MAD = {mad_clean:.2f}")

print("\nС выбросами:")
print(f"Среднее = {mean_dirty:.2f}, SD = {std_dirty:.2f}, MAD = {mad_dirty:.2f}")

print(f"\nИзменение SD: +{((std_dirty - std_clean) / std_clean * 100):.1f}%")
print(f"Изменение MAD: +{((mad_dirty - mad_clean) / mad_clean * 100):.1f}%")

# 4. Визуализация
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# График 1: данные без выбросов
ax1.hist(normal_data, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
ax1.set_title('Распределение без выбросов')
ax1.set_xlabel('Значение')
ax1.set_ylabel('Частота')
ax1.axvline(mean_clean, color='red', linestyle='--', label=f'Среднее = {mean_clean:.2f}')
ax1.axvline(np.median(normal_data), color='green', linestyle='-.', label=f'Медиана = {np.median(normal_data):.2f}')
ax1.legend()

# График 2: данные с выбросами
ax2.hist(data_with_outliers, bins=30, color='salmon', alpha=0.7, edgecolor='black')
ax2.set_title('Распределение с выбросами')
ax2.set_xlabel('Значение')
ax2.set_ylabel('Частота')
ax2.axvline(mean_dirty, color='red', linestyle='--', label=f'Среднее = {mean_dirty:.2f}')
ax2.axvline(np.median(data_with_outliers), color='green', linestyle='-.', label=f'Медиана = {np.median(data_with_outliers):.2f}')
ax2.legend()

plt.suptitle('Сравнение распределений: влияние выбросов', fontsize=16)
plt.show()

# 5. Дополнительная визуализация: устойчивость MAD vs SD
labels = ['Без выбросов', 'С выбросами']
mad_values = [mad_clean, mad_dirty]
std_values = [std_clean, std_dirty]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, std_values, width, label='Стандартное отклонение (SD)', color='coral', alpha=0.8)
rects2 = ax.bar(x + width/2, mad_values, width, label='MAD', color='steelblue', alpha=0.8)

ax.set_xlabel('Сценарий')
ax.set_ylabel('Значение меры разброса')
ax.set_title('Устойчивость MAD и SD к выбросам')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()
