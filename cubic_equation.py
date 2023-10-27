from cmath import *

# https://stackoverflow.com/questions/1829330/solving-a-cubic-equation

def equation_solution_1var(a:float, b:float, c:float, d:float):
    solutions = set()

    def cbrt(polynomial):
        solution = set()
        root1 = polynomial ** (1 / 3)
        root2 = (polynomial ** (1 / 3)) * (-1 / 2 + (sqrt(3) * 1j) / 2)
        root3 = (polynomial ** (1 / 3)) * (-1 / 2 - (sqrt(3) * 1j) / 2)
        solution.update({root1, root2, root3})
        return solution

    def cardano(a, b, c, d):
        p = (3 * a * c - b ** 2) / (3 * a ** 2)
        q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)
        alpha = cbrt(-q / 2 + sqrt((q / 2) ** 2 + (p / 3) ** 3))
        beta = cbrt(-q / 2 - sqrt((q / 2) ** 2 + (p / 3) ** 3))
        for i in alpha:
            for j in beta:
                if abs((i * j) + p / 3) <= 0.00001:
                    x = i + j - b / (3 * a)
                    solutions.add(x)

    def quadratic(a, b, c):
        D = b ** 2 - 4 * a * c
        x1 = (-b + sqrt(D)) / 2 * a
        x2 = (-b - sqrt(D)) / 2 * a
        solutions.update({x1, x2})

    def linear(a, b):
        if a == 0 and b == 0:
            solutions.add("True")
            print('True')

        if a == 0 and b != 0:
            solutions.add("False")
            print('False')

        if a != 0:
            solutions.add(-b / a)

    if a != 0:
        cardano(a, b, c, d)

    elif b != 0:
        quadratic(b, c, d)
    else:
        linear(c, d)
    return solutions

def abcd1():
    # coeff. of equation
    # ax^3+bx^2+cx+d=0
    a:float = 1.0
    b:float = 0.0
    c:float = 0.2 - 1.0
    d:float = -0.7 * 0.2
    return a, b, c, d

def abcd2():
    # coeff. of equation
    # ax^3+bx^2+cx+d=0
    a:float = 0.0
    b:float = 0.0
    c:float = 0.2 - 1.0
    d:float = 0
    return a, b, c, d

def abcd3():
    # coeff. of equation
    # ax^3+bx^2+cx+d=0
    a = 0
    b = 0
    c:float = 0.0
    d:float = -0.7 * 0.2
    return a, b, c, d

def abcd4():
    # coeff. of equation
    # ax^3+bx^2+cx+d=0
    a = 1
    b = 0
    c = -2
    d = -5
    return a, b, c, d

def thetest_equation_solution(a,b, c, d, x):
    # print('Equation ax^3+bx^2+cx+d=0')
    res_f = a*x**3 + b*x**2 + c*x + d
    # print(f'{res_f=}')
    return res_f

def thetest_equation_solution_1var():
    print('ax^3+bx^2+cx+d=0')
    [a, b, c, d] = abcd1()
    solutions = equation_solution_1var(a, b, c, d)
    print(a, b, c, d)
    print(f'{type(solutions)}')
    print(f'{type(list(solutions)[0])}')

    print(f'\nВсего решений = {len(solutions)}')
    print(f'\n{solutions=} \n')

    i = 0
    for res in solutions:
        i+=1
        b = isinstance(res, (float, complex))
        print(f'{i=} ', end=' ')
        if b:
            x = res; print(x)
            res_f= thetest_equation_solution(a, b, c, d, x)
            print(f'{res_f=} \n')
        else:
            print(f' not \n')

if __name__ == "__main__":
    thetest_equation_solution_1var()

