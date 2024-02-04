## --  Вычисление нескольких вариантов полей и построенние их графиков
import matplotlib.pyplot as plt
import math; import copy

# Варьируем h - глубина центра шара в метрах.
# r - радиус шара в метрах, j - намагниченность в А/м

hi = [50.0, 100, 150, 200]  # несколько вариантов глубин
ri = [10.0, 20.0]           # несколько вариантов радиуса
ji = [0.250, 0.500]         # несколько вариантов намагниченности

coo: list = []  # Для вычисления координат пикетов
for i in range(-250, 255, 5):
    coo = coo + [i]
numel: int = len(coo)  # определяем число элементов списка
z_sgs = copy.deepcopy(coo)  # Делаем полную копию списка
h_sgs = copy.deepcopy(coo)  # Делаем полную копию списка
z_si = copy.deepcopy(coo)  # Делаем полную копию списка
h_si = copy.deepcopy(coo)  # Делаем полную копию списка
t_si = copy.deepcopy(coo)  # Делаем полную копию списка
coeff: float = 4*math.pi*math.pow(10, -7)/(4*math.pi)
coeff = coeff*math.pow(10, 9)  # перевод в нанотесла

# задаем размер окна
plt.figure(figsize=(18, 12)) # x и y – ширина и высота рис. в  дюймах
colors = ['red', 'black', 'green']    # цвета для отдельных кривых магнитного поля
linestyles = ['-', '--', '-.', ':']   # типы линий для вариантов сочетаний радиуса и намагниченности

for i in range(len(hi)):             # Перебор по глубинам
    nls_ = -1  # код типа линии сбрасываем для каждого варианта глубины
    plt.subplot(2, 2, i+1)  # 2 - количество строк; 1 - количество столбцов; 1 - индекс ячейки в которой работаем
    plt.title('Магнитное поле шара для глубины '+str(hi[i])+', м', alpha=1, color='b', fontsize=10, fontstyle='italic', fontweight='bold')
    # print('i=', i)
    for k in range(len(ri)):    # Перебор по радиусам
        for l in range(len(ji)):  # Перебор по намагниченностям
            nls_ = nls_ + 1
            m = (4/3)*ji[l]*math.pi*(ri[k]**3)  # магнитный момент шара
            h2 = hi[i]**2;   h22 = 2*h2         # заготовки констант для цикла
            for n in range(numel): # Перебор по точкам профиля
                # print('n=', n)
                x2 = coo[n]**2
                bottom = (h2 + x2)**2.5
                z_sgs[n] = m * (h22 - x2) / bottom         # Вертик. сост. в СГС
                h_sgs[n] = (-3 * m * hi[i] * coo[n]) / bottom  # Гориз. сост. в СГС
                z_si[n] = z_sgs[n]*coeff  # Вертик. сост. в СИ в нТл
                h_si[n] = h_sgs[n]*coeff  # Гориз. сост. в СИ в нТл
                t_si[n] = math.hypot(z_si[n], h_si[n])  # Полный вектор в СИ в нТл

            # Линии c необходимым форматированием
            plt.plot(coo, z_si, color=colors[0], linestyle=linestyles[nls_], label='Верт., r='+str(ri[k])+',м  j='+str(ji[l])+' А/м')
            plt.plot(coo, h_si, color=colors[1], linestyle=linestyles[nls_], label='Гориз., r='+str(ri[k])+',м  j='+str(ji[l])+' А/м')
            plt.plot(coo, t_si, color=colors[2], linestyle=linestyles[nls_], label='Полный, r='+str(ri[k])+',м  j='+str(ji[l])+' А/м \n')

    plt.xlabel('x, м')  # название оси абсцисс
    plt.ylabel('Магнитное поле, нТл')   # название оси ординат
    plt.legend(loc='upper right', fontsize=7)  # положение легенды в верхнем правом углу
    plt.grid()  # включение отображение сетки
plt.show()  # показываем окно с графиками
print('Нормальное завершение')
