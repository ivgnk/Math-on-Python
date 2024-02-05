import sympy as sym
# здесь различий как бы нет
a_s = sym.Rational(1, 2); a_p: float = 0.5
print(a_s,'  ',a_p)
# здесь различия видны
a_s = sym.Rational(1, 3); a_p: float = 1/3
print(a_s,'  ',a_p)
