"""
Задача 6. Оценка качества продукции
P_defect = 0.05 (брак) - вероятность брака
P_reject_given_defect = 0.9 (выявление брака).
P_reject_given_good = 0.02 (ложное отбраковывание).
Найти: P_defect_given_reject — вероятность брака, если изделие отбраковано.
"""
P_defect = 0.05
P_good = 0.95
P_reject_given_defect = 0.9
P_reject_given_good = 0.02

P_reject = P_reject_given_defect * P_defect + P_reject_given_good * P_good
P_defect_given_reject = (P_reject_given_defect * P_defect) / P_reject

print(f"P(Defect | Reject) = {P_defect_given_reject:.4f}")  # ≈ 0.7031


