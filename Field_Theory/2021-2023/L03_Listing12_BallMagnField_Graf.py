# Параметры исходного шара
# h - глубина центра шара в метрах
# r - радиус шара в метрах
# j - намагниченность в А/м
import math; import copy
import matplotlib.pyplot as plt

h: float = 50.0
r: float = 25
j: float = 0.5

# Вычисляем магнитный момент шара
m:float = (4/3)*j*math.pi*(r**3)
# Вычисляем прочие константы
h2 = h**2;   h22 = 2*h2

z_sgs: list = [] # готовим пустой список элементов для вертик.составл.
h_sgs: list = [] # готовим пустой список элементов для гориз.составл.

# профиль с пикетами от -250 м до +250 м, шаг 5 м.
coo: list = []
for i in range(-250,255,5):
    coo = coo + [i]

for x in coo:
    x2 = x**2 # заготовка элемента общего для z и h
    bottom:float = (h2+x2)**2.5 # заготовка знаменателя
    z_sgs = z_sgs + [m*(h22-x2)/bottom]
    h_sgs = h_sgs + [-3*m*h*x/bottom]

# Подготовка списков для вычисления в СИ
z_si = copy.deepcopy(z_sgs) # Делаем полную копию списка
h_si = copy.deepcopy(h_sgs) # Делаем полную копию списка
# Непосредственно сам цикл вычисления в СИ
numel:int = len(z_sgs) # определяем число элементов списка
i: int = 0 # счетчик
coeff: float = 4*math.pi*math.pow(10,-7)/(4*math.pi)
coeff = coeff*math.pow(10,9) # перевод в нанотесла
while i < numel:
    z_si[i] = z_sgs[i]*coeff
    h_si[i] = h_sgs[i]*coeff
    i = i + 1

# Делаем полную копию списка для полного вектора
t_si = copy.deepcopy(h_sgs)

# вычисление корня из суммы квадратов указанных элементов
for i in range(numel):
    t_si[i]= math.hypot(z_si[i], h_si[i])
    # print(i, coo[i], z_si[i], h_si[i], t_si[i])

# Построение графиков
plt.plot(coo, z_si,  label = 'Верт.компонента')
plt.plot(coo, h_si,  label = 'Гориз.компонента')
plt.plot(coo, t_si,  label = 'Полный вектор')
plt.title('Магнитное поле шара', alpha=1, color='r', fontsize=18, fontstyle='italic', fontweight='bold')

# положение легенды в верхнем правом углу
plt.legend(loc='upper right')
plt.grid()             # включение отображение сетки
plt.xlabel('x, м')     # название оси абсцисс
plt.ylabel('Магнитное поле, нТл') # название оси ординат
plt.show()      # показываем график