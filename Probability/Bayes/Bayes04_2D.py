"""
Задача 4. Диагностика оборудования
P_failure = 0.02 (вероятность сбоя).
P_alarm_given_failure = 0.98 (вероятность сигнала при сбое).
P_alarm_given_no_failure = 0.01 (вероятность ложной тревоги).

Найти: P_failure_given_alarm — вероятность сбоя при сигнале.
"""
# P_failure = 0.02
import numpy as np
import matplotlib.pyplot as plt
##-1-beg---------
P_failure_min=0.02
P_failure_max=0.04
n_failure = 80
P_failure_arr=np.linspace(P_failure_min,P_failure_max,n_failure)
##-1-end---------
##-2-beg---------
P_alarm_given_failure_min = 0.80
P_alarm_given_failure_max = 0.90
n_alarm_given_failure = 85
P_alarm_given_failure_arr = np.linspace(P_alarm_given_failure_min,P_alarm_given_failure_max,n_alarm_given_failure)
##-2-end---------
P_alarm_given_no_failure = 0.01

res_arr=np.zeros((n_failure, n_alarm_given_failure))

for i_failure in range(n_failure):
    P_failure=P_failure_arr[i_failure]
    P_no_failure = 1 - P_failure
    for i_alarm_given_failure in range(n_alarm_given_failure):
        P_alarm_given_failure=P_alarm_given_failure_arr[i_alarm_given_failure]
        P_alarm = P_alarm_given_failure * P_failure + P_alarm_given_no_failure * P_no_failure
        P_failure_given_alarm = (P_alarm_given_failure * P_failure) / P_alarm
        print(f"{P_failure:12f} {P_alarm_given_failure:12f}  {P_failure_given_alarm:12f}")
        res_arr[i_failure,i_alarm_given_failure]=P_failure_given_alarm

plt.figure(figsize=(8, 6))
plt.imshow(res_arr,origin='lower',cmap=plt.get_cmap('jet'),
           # значения по осям
           extent=[P_failure_min, P_failure_max, P_alarm_given_failure_min,P_alarm_given_failure_max],
           # растягивает под квадрат
           aspect='auto'
           )
plt.xlabel("P_failure (вероятность сбоя)")
plt.ylabel("P_alarm_given_failure (вероятность сигнала при сбое)")
plt.grid()

plt.title('P_failure_given_alarm (вероятность сбоя при сигнале)')
plt.colorbar(label='Значение вероятности', drawedges=False)
plt.show()
