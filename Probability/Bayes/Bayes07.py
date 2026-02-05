"""
Задача 7. Прогноз погоды
P_rain = 0.3 (вероятность дождя).
P_forecast_rain_given_rain = 0.8 (прогноз при дожде).
P_forecast_rain_given_no_rain = 0.1 (ложный прогноз).
Найти: P_rain_given_forecast — вероятность дождя, если прогноз говорит «дождь».
"""

P_rain = 0.3
P_no_rain = 0.7
P_forecast_rain_given_rain = 0.8
P_forecast_rain_given_no_rain = 0.1

P_forecast_rain = (P_forecast_rain_given_rain * P_rain +
                   P_forecast_rain_given_no_rain * P_no_rain)
P_rain_given_forecast = (P_forecast_rain_given_rain * P_rain) / P_forecast_rain

print(f"P(Rain | Forecast Rain) = {P_rain_given_forecast:.4f}")  # ≈ 0.7742


