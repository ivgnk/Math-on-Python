# https://www.youtube.com/watch?v=GVR3dp97d20

# Найти минимальное натуральное N такое что
# N+2018 делится на 2020, а
# N+2020 делится на 2018

# Find the minimum natural N such that
# N+2018 is divisible by 2020,
# and N+2020 is divisible by 2018

for i in range(0,3_000_000):
    a1 = i+2018
    a2 = i+2020
    r1 = a1 % 2020
    r2 = a2 % 2018
    if (r1 == 0) and (r2 == 0):
        print(f'Result {i=}')   # Result i=2034142
        break
    # else: print(f'{i:4}          {r1:8}       {r2:8}')
