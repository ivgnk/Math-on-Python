# Задание комплексных матриц
import numpy as np
print('\n Первый способ')
c = np.zeros((2,2),dtype=complex)
print(c)
c[0,0] = 1+1j
print(c)

print('\n Второй способ')
c = np.zeros((2,2),dtype=np.complex_)
print(c)
c[0,0] = 1+1j
print(c)

print('\n Третий способ')
import numpy as np
c2 = np.empty([2,2]);     print(c2)
c2 = c2.astype(complex);  print(c2)
c2[0,0] = 5j+2;             print(c2)
c2[0] = 5j+2;             print(c2)
c2[1] = 7j+1;             print(c2)

# Умножение комплексных матриц
import numpy as np
print('\n Умножение комплексных чисел')
a = 2 + 3j; b = 8 + 7j
print('(a+bi)*(c+di) = (ac-db) + (bc+ad)i = ',(2*8-3*7)+1j*(3*8+2*7))
print('a*b = ',a*b)
print('np.dot(a*b) = ',np.dot(a, b))
print('np.vdot(a*b) = ',np.vdot(a, b))
print('np.vdot(a*b)/in/ = ', a.conjugate()*b)

print('\n Умножение комплексных матриц')
import numpy as np
x0 = np.array([2, 4]); y0 = np.array([8, 5]);
print('np.dot(x0, y0) = ',np.dot(x0, y0))
print('np.vdot(x0, y0) = ',np.vdot(x0, y0))
print('dot/vdot(x0, y0)/in/ = ',2*8+4*5)
x1 = np.array([2 + 3j, 4 + 5j]);
y1 = np.array([8 + 7j, 5 + 6j]);
x2 = np.conjugate(x1)
# vdot() -скалярное произведение векторов a и b
print('np.dot(x1, y1) = ',np.dot(x1, y1))
print('np.vdot(x1, y1) = ', np.vdot(x1, y1))
print('np.vdot(x1, y1) /in/= ', np.dot(x2, y1))

