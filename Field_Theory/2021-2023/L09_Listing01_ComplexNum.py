z = complex(7, -8)
print(z, type(z))

z = 4+3j
print(z, type(z))

z1 = 4+3J
print(z1, type(z1))

z = complex("4+3j")
print(z, type(z))

z1 = complex(); z2 = complex(0); z3 = complex(3)
print(z1, z2, z3)
z1 = complex('2'); z2 = complex('2j'); z3 = 4+0j; z4 = 2j
print(z1, z2, z3, z4)
print(type(z1), type(z2), type(z3), type(z4))

z = -3+7j
print(z.real, z.imag)

z = -3+7j
print(z.conjugate())

a = 1 + 2j; b = 2 + 4j
print('Сложение =', a + b)
print('Вычитание =', a - b)
print('Умножение =', a * b)
print('Деление =', a / b)

a = 1 + 2j; b = 2 + 4j; c = 1 + 2j;
print(a == b, a == c) # проверка равенства
print(a != b, a != c) # проверка неравенства

print(abs(a))

print('Возведение в степень')
a = 1 + 2j; a2 = a*a; a3 = a*a*a; a4 = a*a*a*a;
# сравнение двух способов
print(a**2, a2); print(a**3, a3); print(a**4, a4)
print(); print(a);
print(a2**0.5); print(a3**(1/3));
r4=a4**0.25
print(r4,'     ',r4**4)

