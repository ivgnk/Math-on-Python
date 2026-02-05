"""
Задача 4. Диагностика оборудования
P_failure = 0.02 (вероятность сбоя).
P_alarm_given_failure = 0.98 (сигнал при сбое).
P_alarm_given_no_failure = 0.01 (ложная тревога).

Найти: P_failure_given_alarm — вероятность сбоя при сигнале.
"""
P_failure = 0.02
P_no_failure = 0.98
P_alarm_given_failure = 0.98
P_alarm_given_no_failure = 0.01

P_alarm = P_alarm_given_failure * P_failure + P_alarm_given_no_failure * P_no_failure
P_failure_given_alarm = (P_alarm_given_failure * P_failure) / P_alarm

print(f"P(Failure | Alarm) = {P_failure_given_alarm:.4f}")  # ≈ 0.6689

