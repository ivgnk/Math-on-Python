"""
Задача 4. Диагностика оборудования
P_failure = 0.02 (вероятность сбоя).
P_alarm_given_failure = 0.98 (вероятность сигнала при сбое).
P_alarm_given_no_failure = 0.01 (вероятность ложной тревоги).

Найти: P_failure_given_alarm — вероятность сбоя при сигнале.
"""
import numpy as np
import matplotlib.pyplot as plt

P_failure_min=0.01
P_failure_max=0.40
n_failure = 200
P_failure_arr=np.linspace(P_failure_min,P_failure_max,n_failure)   # реально нужны
res=np.linspace(P_failure_min,P_failure_max,n_failure) # просто заготовка под массив
P_alarm_given_failure = 0.98
P_alarm_given_no_failure = 0.01

for i in range(n_failure):
    P_failure=P_failure_arr[i]
    P_no_failure = 1 - P_failure
    P_alarm = P_alarm_given_failure * P_failure + P_alarm_given_no_failure * P_no_failure
    P_failure_given_alarm = (P_alarm_given_failure * P_failure) / P_alarm
    res[i]=P_failure_given_alarm

plt.title("График вероятности сбоя при сигнале")
plt.xlabel("Вероятность сбоя"); plt.ylabel("Вероятность сбоя при сигнале")
plt.plot(P_failure_arr,res)
plt.grid(); plt.show()


