# # lstm_forecast_page.py
# import keras
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# import requests
# import os
# import joblib
#
# def lstm_forecast_page():
#     st.title("🔮 Previsione Consumo Energetico con LSTM")
#
#     # --- Inserisci qui la tua OpenWeather API Key ---
#     api_key = "3001a620158131c941d011ba003258b3"
#
#     # --- Selezione città e orizzonte temporale ---
#     st.subheader("📍 Seleziona città e periodo di previsione")
#     city = st.text_input("Città", value="Milano")
#     forecast_hours = st.slider("⏱ Ore di previsione futura", min_value=1, max_value=48, value=12)
#
#     # --- Carica e pre-processa il dataset locale per training ---
#     @st.cache_data
#     def load_data():
#         data = pd.read_csv("file_ripulito.csv")
#         data['datetime'] = pd.to_datetime(data['datetime'])
#         data.set_index('datetime', inplace=True)
#         return data[['EnergyConsumption', 'Humidity', 'Temperature']].dropna()
#
#     data = load_data()
#     data_hourly = data.resample('1H').mean()
#
#     # --- Normalizzazione ---
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(data_hourly)
#
#     # --- Creazione dataset ---
#     def create_dataset(dataset, look_back):
#         X, y = [], []
#         for i in range(look_back, len(dataset)):
#             X.append(dataset[i - look_back:i])
#             y.append(dataset[i, 0])  # solo EnergyConsumption
#         return np.array(X), np.array(y)
#
#     look_back = 24  # 24 ore (1 giorno)
#     X, y = create_dataset(scaled_data, look_back)
#     split = int(len(X) * 0.8)
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]
#
#     # --- Carica o allena il modello ---
#     model_path = "lstm_model_generic.keras"
#     if os.path.exists(model_path):
#         model = keras.models.load_model(model_path)
#     else:
#         model = Sequential([
#             LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
#             Dropout(0.2),
#             LSTM(50),
#             Dropout(0.2),
#             Dense(25),
#             Dense(1)
#         ])
#         model.compile(optimizer='adam', loss='mean_squared_error')
#         model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=0)
#         model.save(model_path)
#
#     # --- Valutazione del modello ---
#     y_pred = model.predict(X_test)
#     y_pred_inv = scaler.inverse_transform(np.hstack((y_pred, np.zeros((len(y_pred), 2)))))[:, 0]
#     y_test_inv = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), 2)))))[:, 0]
#     rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
#     mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100
#     r2 = r2_score(y_test_inv, y_pred_inv)
#
#     st.markdown("### 🧮 Metriche del modello")
#     st.write(f"**RMSE:** {rmse:.2f}")
#     st.write(f"**MAPE:** {mape:.2f}%")
#     st.write(f"**R²:** {r2:.4f}")
#
#     # --- Relazioni tra feature e target ---
#     st.subheader("📊 Correlazione Umidità / Temperatura con Consumo Energetico")
#     fig_corr, ax_corr = plt.subplots(1, 2, figsize=(12, 4))
#     ax_corr[0].scatter(data_hourly['Temperature'], data_hourly['EnergyConsumption'], alpha=0.3)
#     ax_corr[0].set_title("Temperatura vs Consumo")
#     ax_corr[1].scatter(data_hourly['Humidity'], data_hourly['EnergyConsumption'], alpha=0.3)
#     ax_corr[1].set_title("Umidità vs Consumo")
#     st.pyplot(fig_corr)
#
#     # --- Previsione futura usando l’API OpenWeather ---
#     if st.button("🔄 Prevedi consumo energetico futuro"):
#         st.subheader("🔍 Previsione basata su OpenWeather")
#
#         url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
#         response = requests.get(url)
#         if response.status_code != 200:
#             st.error("Errore nella richiesta API. Verifica il nome della città o la tua API key.")
#             return
#
#         forecast_json = response.json()
#
#         future_features = []
#         timestamps = []
#         for entry in forecast_json['list']:
#             dt_txt = entry['dt_txt']
#             temp = entry['main']['temp']
#             hum = entry['main']['humidity']
#             timestamps.append(dt_txt)
#             future_features.append([temp, hum])
#
#             if len(future_features) >= forecast_hours:
#                 break
#
#         # Costruzione input LSTM per previsione
#         last_known = scaled_data[-look_back:].copy()
#         future_consumptions = []
#
#         for t in range(len(future_features)):
#             temp, hum = future_features[t]
#             next_input = np.append(last_known[1:], [[0, hum, temp]], axis=0)  # 0 perché il consumo è ignoto
#             pred_scaled = model.predict(np.array([next_input]))[0][0]
#             last_known = np.append(last_known[1:], [[pred_scaled, hum, temp]], axis=0)
#             future_consumptions.append(pred_scaled)
#
#         # Inversa normalizzazione delle previsioni
#         future_consumptions_inv = scaler.inverse_transform(np.hstack((
#             np.array(future_consumptions).reshape(-1, 1),
#             np.zeros((len(future_consumptions), 2))
#         )))[:, 0]
#
#         # --- Grafico finale ---
#         future_timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps[:forecast_hours]]
#         st.subheader("📈 Previsione Consumo Energetico Oraria")
#         fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
#         ax_forecast.plot(future_timestamps, future_consumptions_inv, marker='o')
#         ax_forecast.set_xlabel("Data e Ora")
#         ax_forecast.set_ylabel("Consumo Energetico (Watt)")
#         ax_forecast.set_title(f"Previsione Consumo a {city} per {forecast_hours}h")
#         ax_forecast.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d-%m %H:%M'))
#         fig_forecast.autofmt_xdate()
#         st.pyplot(fig_forecast)
