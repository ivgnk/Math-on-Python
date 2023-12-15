'''
from https://docs.sympy.org/latest/guides/solving/reduce-inequalities-algebraically.html

Note: SymPy can currently reduce for only one symbol (variable) in an inequality.

To reduce for more than one symbol in an inequality, try SciPy’s linprog()
To reduce Boolean expressions, use as_set
'''

import inspect
from sympy import Symbol, symbols, reduce_inequalities, pi, sqrt

def thetest_symbols():
    # Difference between sympy.symbols and sympy.Symbol
    # https://stackoverflow.com/questions/62392884/difference-between-sympy-symbols-and-sympy-symbol
    # Symbol is a class.
    # symbols is a function that can create multiple instances of the Symbol class.
    # In [9]: symbols('A B C')
    # Out[9]: (A, B, C)
    #
    # In [10]: Symbol('A B C')
    # Out[10]: A B C
    print('\n',inspect.currentframe().f_code.co_name)
    a = symbols('A')
    b = Symbol('b')
    print(type(a),type(b))
    print('- - -')
    a1 = symbols('A B C')
    b1 = Symbol('a b c')
    print(f'{a1=}  {type(a1)=}')
    print(f'{b1=}      {type(b1)=}')


def Reducing_a_System_of_Inequalities_for_a_Single_Variable_Algebraically():
    print('\n',inspect.currentframe().f_code.co_name)
    x = symbols('x')
    a = reduce_inequalities([x >= 0, x ** 2 <= pi], x) #  (0 <= x) & (x <= sqrt(pi))
    print(a)

if __name__ == "__main__":
    # thetest_symbols()
    Reducing_a_System_of_Inequalities_for_a_Single_Variable_Algebraically()