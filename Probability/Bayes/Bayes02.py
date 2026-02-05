"""
Задача 2. Две урны с шарами
Урна 1: 3 белых, 1 чёрный (P_white_urn1 = 3/4).
Урна 2: 1 белый, 3 чёрных (P_white_urn2 = 1/4).
Выбор урны равновероятен: P_urn1 = P_urn2 = 0.5.

Найти: P_urn1_given_white — вероятность,
что это урна 1, если вынут белый шар.
"""

P_urn1 = 0.5
P_urn2 = 0.5
P_white_urn1 = 3 / 4
P_white_urn2 = 1 / 4

P_white = P_white_urn1 * P_urn1 + P_white_urn2 * P_urn2
P_urn1_given_white = (P_white_urn1 * P_urn1) / P_white

print(f"P(Urn1 | White) = {P_urn1_given_white:.4f}")  # = 0.75

