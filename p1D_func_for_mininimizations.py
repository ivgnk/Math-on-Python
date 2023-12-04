lst_func = []

# ----- Eq011



def for_eq011f(x):
    return 2**x + x - 5

def getEq011_f()->tuple:
    nnn = 100
    name_ = 'function 2**x + x = 5'
    return (nnn, for_eq011f, name_)

# ----- Eq012
def for_eq012f(x):
    return (3+x)/2022 + (2+x)/2023 + (1+x)/2024 + x/2025 + 4

def getEq012_f()->tuple:
    nnn = 3300
    name_ = 'function (3+x)/2022 + (2+x)/2023 + (1+x)/2024 + x/2025 = -4'
    return (nnn, for_eq012f, name_)

def create_function_list():
    lst_func.append(getEq011_f())
    lst_func.append(getEq012_f())
    print(lst_func)
