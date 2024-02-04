#Listing 06_cmath01
print('\n Вычисление модуля')
import cmath; import math
num = 4 + 3j
p = cmath.phase(num); print('cmath Module:', p)
p2 = math.atan(num.imag/num.real); print('Math Module:', p2)

#Listing 06_cmath02
print('\n Преобразование в полярные и прямоугольные координаты')
import cmath
a = 3 + 4j
polar_coordinates = cmath.polar(a); print('polar = ', polar_coordinates)
modulus = abs(a)
phase = cmath.phase(a)
rect_coordinates = cmath.rect(modulus, phase); print('rect = ',rect_coordinates)

#Listing 06_cmath03
print('\n Тригонометрические функции')
import cmath; a = 3 + 4j
print('Sine:', cmath.sin(a))
print('Cosine:', cmath.cos(a))
print('Tangent:', cmath.tan(a))
print('ArcSin:', cmath.asin(a))
print('ArcCosine:', cmath.acos(a))
print('ArcTan:', cmath.atan(a))

#Listing 06_cmath04
print('\n Гиперболические функции')
import cmath; a = 3 + 4j
print('Hyperbolic Sine:', cmath.sinh(a))
print('Hyperbolic Cosine:', cmath.cosh(a))
print('Hyperbolic Tangent:', cmath.tanh(a))
print()
print('Inverse Hyperbolic Sine:', cmath.asinh(a))
print('Inverse Hyperbolic Cosine:', cmath.acosh(a))
print('Inverse Hyperbolic Tangent:', cmath.atanh(a))

#Listing 06_cmath05
print('\n Функции классификации')
print('Конечное ', cmath.isfinite(2 + 2j))
print('Конечное ',cmath.isfinite(cmath.inf + 2j))
print('Бесконечное ',cmath.isinf(2 + 2j))
print('Бесконечное ',cmath.isinf(cmath.inf + 2j))
print('Бесконечное ',cmath.isinf(cmath.nan + 2j))
print('NaN ',cmath.isnan(2 + 2j))
print('NaN ',cmath.isnan(cmath.inf + 2j))

#Listing 06_cmath05
print('\n Функция близости')
print(cmath.isclose(2+2j, 2.01+1.9j, rel_tol=0.05))
print(cmath.isclose(2+2j, 2.01+1.9j, abs_tol=0.005))
