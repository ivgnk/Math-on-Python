# Параметры исходного шара
# h - глубина центра шара в метрах
# r - радиус шара в метрах
# j - намагниченность в А/м
import math; import copy
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