import random
import matplotlib.pyplot as plt
import numpy as np
# Функция  y = x+x2-x3 на отрезке [-10,10], вычисляется с шагом 0.1.
# И добавляется шум по формуле 0.5*y*случайное вещественное число от -1 до 1

x = np.arange(-10.0, 10.1, 0.2) # т.к. действительные числа области определения
y = -x + x**2 - x**3
random.seed(123)  # 	т.к. хотим чтобы случайные последовательности всегда были одинаковы

# т.к. умножаем каждый элемент списка на число нужен массив numpy
rnd = np.array([random.uniform(-1,1) for i in range(len(x))])
noise1 = 0.75*y*rnd;         y1 = y + noise1
noise2 = 0.75*np.max(y)*rnd; y2 = y + noise2

# plt.plot(x, y, label ='Исход.' ); plt.plot(x, y1, label ='+шум1' );
# plt.plot(x, y2, label ='+шум2' )
# plt.legend(loc='upper right'); plt.grid();
# plt.show()

# можно также использовать np.average
print('Среднее =', np.mean(y), np.mean(y1), np.mean(y2))
print('Медиана =', np.median(y), np.median(y1), np.median(y2))
a = np.mean(y);    b = np.mean(y1);   c = np.mean(y2)
a1 = np.median(y); b1 = np.median(y1); c1 = np.median(y2)
print('g')
print(f"Среднее = {a:14g}  {b:14g}  {c:14g}")
print(f"Медиана = {a1:14g}  {b1:14g}  {c1:14g}")
print('f умолч')
print(f"Среднее = {a:f}  {b:f}  {c:f}")
print(f"Медиана = {a1:f}  {b1:f}  {c1:f}")
print('f 12')
print(f"Среднее = {a: 12f}  {b: 12f}  {c: 12f}")
print(f"Медиана = {a1: 12f}  {b1: 12f}  {c1: 12f}")
print('f 12.4')
print(f"Среднее = {a: 12.4f}  {b: 12.4f}  {c: 12.4f}")
print(f"Медиана = {a1: 12.4f}  {b1: 12.4f}  {c1: 12.4f}")

dat = np.array([y,y1,y2])
label_ =np.array(['Исход.', '+шум1','+шум2'])
mean_all = np.array([a, b, c])
summ = np.array([0.0, 0.0, 0.0])
n = len(x)
for i in range(3):
    print(label_[i])
    for j in range(n):
        aa = dat[i,j]-mean_all[i]
        summ[i] = summ[i]+np.power(aa, 2)
    print('Станд.Откл выборки = ', np.sqrt(summ[i]/(n-1)))
    print('Станд.Откл ген.совок = ', np.sqrt(summ[i]/n))

for i in range(3):
    print(i,' np.std = ',np.std(dat[i]))

for i in range(3):
    print(i,' np.std (ddof=1) =',np.std(dat[i],ddof=1))
print('var')
for i in range(3):
    print(i,' np.var    (ddof=0) =',np.var(dat[i],ddof=0))
    print(i,' np.std**2 (ddof=0) =',np.std(dat[i],ddof=0)**2)
print('stats.mode')
from scipy import stats
for i in range(3):
    print(i,' stats.mode =',stats.mode(dat[i]))

import matplotlib.pyplot as plt
plt.figure(figsize=(27, 9))
for i in range(3):
    plt.subplot(1, 3, i+1) # 2 - количество строк; 1 - количество столбцов; 1 - индекс ячейки в которой работаем
    plt.title(label_[i])
    counts, bins = np.histogram(dat[i])
    plt.hist(dat[i], density=False, bins=len(counts))  # density=False would make counts
    plt.grid()
    plt.ylabel('Numbers')
plt.show()

print('counts, bins ')
for i in range(3):
    counts, bins = np.histogram(dat[i])
    # Два последних  числа должны совпадать
    print(label_[i], len(dat[i]), np.sum(counts))
    print(counts, bins)

import matplotlib.pyplot as plt
plt.figure(figsize=(35, 12))
plt.title('Разные варианты задания гистограммы')
for j in range(2):
    for i in range(3):
        plt.subplot(2, 3, i+1+j*3) # 2 - количество строк; 3 - количество столбцов; индекс ячейки в которой работаем
        plt.title(label_[i])
        if j==0:   # принудительно задали 30 столбцов
            plt.hist(dat[i], density=False, bins = 30) # , bins = 30
        else:
            counts, bins = np.histogram(dat[i])
            plt.hist(dat[i], density=False, bins=len(counts))
        plt.grid()
        plt.ylabel('Numbers')
plt.show()

print('Формула Стерджесса')
NN = len(dat[1])
print(np.log10(10))
nint = 1 + round(3.322*np.log10(NN))
print(nint)

print('Правило Фридмана – Дикона')
for i in range(3):
    q25, q75 = np.percentile(dat[i], [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
    bins = round((np.max(dat[i]) - np.min(dat[i])) / bin_width)
    print(f"{label_[i]:7s}  {bins:14g}")



print("           Min             Max        Interval")
for i in range(3):
    mi = np.min(dat[i]); ma = np.max(dat[i]); intr = ma-mi
    print(f"{mi:14g}  {ma:14g}  {intr:14g}")

print("         Сумма           Число")
for i in range(3):
    sum = np.sum(dat[i]); num = len(dat[i])
    print(f"{sum:14g}  {num:14g}")

print("Данные  Квартиль    Персентиль        Значение")
quart   = np.array([0, 1, 2, 3, 4])
percent = np.array([0, 25, 50, 75, 100])
for i in range(3):
    for j in range(len(quart)):
        print(f"{label_[i]:7s} {quart[j]:4g} {percent[j]:14g}%  {np.percentile(dat[i],percent[j]):14g}")



from scipy.stats import skew
print('Номер  Асимметрия корр.  Асимметрия некорр.')
for i in range(3):
    skew_corr = skew(dat[i], bias=True) # скорректированное за статист. смещение
    skew_notcorr = skew(dat[i], bias=False) # НЕскорректированное за статист. смещение
    print(f"{i:4g}  {skew_corr:14g} {skew_notcorr:14g}")

from scipy.stats import kurtosis
print('Номер  Эксцесс корр.  Эксцесс некорр.')
for i in range(3):
    kurt_corr = kurtosis(dat[i], bias=True) # скорректированное за статист. смещение
    kurt_notcorr = kurtosis(dat[i], bias=False) # НЕскорректированное за статист. смещение
    print(f"{i:4g}  {kurt_corr:14g} {kurt_notcorr:14g}")

