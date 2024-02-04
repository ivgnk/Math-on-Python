import numpy as np
'''
Работа с двумерными матрицами 
'''

print('Матричное умножение')
a = np.arange(1, 7, 1).reshape(2, 3); print(a)
b = np.arange(10, 130, 10).reshape(3, 4); print(b)
c = np.dot(a, b); print(c)

print('\n Некоммутативность матричного умножения')
a=np.array([[1,2],[3,4]])
b=np.array([[5,1],[-1,0]])
print(np.dot(a, b))
print(np.dot(b, a))
c = np.dot(a, a); print(np.dot(c, a))

print('\n Транспонирование матриц')
b = np.arange(10, 130, 10).reshape(3, 4)
print (a.transpose())
print (b.transpose())

print('\n Сложные матричные вычисления')
a = np.array([[1,0,-1],[2,3,-2]])
b = np.array([[5,1],[3,2],[4,-3]])
c = np.array([[2,0,-1,3],[5,1,2,5]])
d = np.array([[7,1,1,-3],[4,1,-2,0]])
print(np.dot(a.transpose()-2*b, 2*c-5*d))

print('\n Разделение матрицы на строки')
a = np.arange(10, 130, 10).reshape(3, 4); print(a)
print('Используем vsplit')
c = np.vsplit(a,3)
print(c, type(c), type(c[0]))
print()
print('Используем срезы')
for i in range(3):
    b = a[i,:]
    print(b, type(b))

print('\n Вычисляем определитель матрицы')
import numpy as np
from numpy import linalg as ln
a=np.array([[4,6,-2,4],[1,2,-3,1],[4,-2,1,0],[6,4,4,6]])
print(a)
det_A=ln.det(a)
print(det_A)

print('\n Вычисляем обратную матрицу')
inv_a=ln.inv(a)
print(inv_a)          #первая обратная матрица
print(ln.inv(inv_a))  #вторая обратная матрица, д.б. равна исходной
#Второй вариант вычисления второй обратной матрицы
inv_a1 = np.copy(inv_a)
inv_a1[abs(inv_a1)<1e-9] = 0.0 # Зануляем близкие к нулю значения
print(inv_a1)
print(ln.inv(inv_a1)) #еще раз считаем обратную матрицу, чтобы сравнить с исходной

print('\n Решить матричное уравнение')
import numpy as np
from numpy import linalg as ln
a = np.array([[3,1],[-3,1]])
b = np.array([[9,5],[-3,-1]])
print(np.dot(ln.inv(a), b))

print('\n №1 Решить квадратную СЛАУ ')
import numpy as np
a = np.array([[2,1,-2],[1,-2,1],[3,1,-1]])
b = np.array([-3,5,0])
x1 = np.linalg.solve(a, b)
x2 = np.dot(np.linalg.inv(a), b)
print('№1 Решение СЛАУ'); print(x1)
print('№2 Решение СЛАУ'); print(x2)
print('Проверка')
print(np.dot(a, x1))

print('\n №2 Решить квадратную СЛАУ ')
a = np.array([[3,-1,1],[1,-1,-1],[5,-3,-1]]);
b = np.array([5,2,10])
x1 = np.linalg.solve(a, b); print(x1)
x2 = np.dot(np.linalg.inv(a), b); print(x2)
print('Проверка')
print(np.dot(a, x1))
print(np.dot(a, x2))

print('Равенство х1 и х2 = ', np.array_equiv (x1, x2))
print('Разность решений x1 и x2 = ', x1-x2)
# ответ согласно книге
x3 = np.array([-1.2857142857142854, -2.142857142857143, 6.7142857142857135])
print('Постановка книжного ответа дает св.член = ', np.dot(a, x3))
print('Детерминант матрицы = ', np.linalg.det(a))

print('\n №3 Решить квадратную СЛАУ ')
from numpy import linalg as ln
A=np.array([[3,-1,1],[1,-1,-1],[5,-3,-1]])
B=np.array([3,11,8])
x1 = np.linalg.solve(A,B)
print(x1)
Det_A=ln.det(a)
print (Det_A)

print('\n №4 Решить квадратную СЛАУ по правилу Крамера')
a =np.array([[2,1,-2],[1,-2,1],[3,1,-1]])
b =np.array([-3,5,0])
print(ln.det(a))
x1 = np.linalg.solve(a, b); print(x1)
x2 = np.dot(np.linalg.inv(a), b); print(x2)

print('\n №5 используем теорему Кронекера-Капелли')
a = np.array([[3,-1,1],[1,-1,-1],[5,-3,-1]]);
b = np.array([5,2,10])
b1=b.reshape(1,b.shape[0])
print(b)
print(b1)
ab=np.vstack((a,b1))
print(ab)
rank_a=np.linalg.matrix_rank(a); print(rank_a)
rank_ab=np.linalg.matrix_rank(ab); print(rank_ab)
if rank_a==rank_ab:
    print ('Система имеет бесконечное множество решений, частное решение +X')
else:
    print ('Система не имеет решений')

print('\n №6 используем теорему Кронекера-Капелли')
a = np.array([[3,-1,1],[1,-1,-1],[5,-3,-1]])
b = np.array([5,2,10])
ab1 = np.c_[a, b] # объединение матриц по столбцам, с_ - "column"
print(a); print(ab1)
rank_a = np.linalg.matrix_rank(a); print('rank_a = ',rank_a)
rank_ab1 = np.linalg.matrix_rank(ab1); print('rank_ab1 = ', rank_ab1)
if rank_a == rank_ab1:
    print('Система имеет бесконечное множество решений')
else:
    print ('Система не имеет решений')

print('\n №7 используем теорему Кронекера-Капелли')
a = np.array([[-6,9,3,2],[-2,3,5,4],[-4,6,4,3]])
b = np.array([4,2,3])
ab1 = np.c_[a, b] # объединение матриц по столбцам, с_ - "column"
print(a); print(ab1)
rank_a = np.linalg.matrix_rank(a); print('rank_a = ',rank_a)
rank_ab1 = np.linalg.matrix_rank(ab1); print('rank_ab1 = ', rank_ab1)
if rank_a == rank_ab1:
    print('Система имеет бесконечное множество решений')
    x=np.linalg.solve(a, b)
    print('Одно из решений')
    print(x)
else:
    print ('Система не имеет решений')
