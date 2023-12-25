'''
15 авг 2022 Эффективное использование any и all в Python
https://habr.com/ru/companies/wunderfund/articles/681426/
'''
import  numpy as np
def all_not_equal_to_200_million() -> bool:
    return all(False for number in range(1_000_000_000) if number == 200_000_000)

def all_bigger_than0(x):
    # https://habr.com/ru/articles/320288/
    z = [True if number > 0 else False for number in x]
    return z, all(z)


# print(all_not_equal_to_200_million())
x = np.linspace(-5,5,201)
print(x)
y = 
print(f'{all_bigger_than0(x)=}')