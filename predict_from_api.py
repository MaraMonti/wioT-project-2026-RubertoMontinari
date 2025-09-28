import streamlit as st
import requests
import datetime
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# def get_weather_data(city_name, start_date, end_date, api_key):
#     url = f"https://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={api_key}&units=metric"
#     response = requests.get(url)
#     response.raise_for_status()
#     data = response.json()
#
#     temps = []
#     humidities = []
#     dates = []
#     hours = []
#     dayofweeks = []
#
#     for item in data.get('list', []):
#         dt_txt = item['dt_txt']
#         dt = datetime.datetime.strptime(dt_txt, "%Y-%m-%d %H:%M:%S")
#
#         if start_date <= dt.date() <= end_date:
#             temps.append(item['main']['temp'])
#             humidities.append(item['main']['humidity'])
#             dates.append(dt.date())
#             hours.append(dt.hour)
#             dayofweeks.append(dt.weekday())  # 0=Lunedì, 6=Domenica
#
#     return pd.DataFrame({
#         'Date': dates,
#         'Temperature': temps,
#         'Humidity': humidities,
#         'Hour': hours,
#         'DayOfWeek': dayofweeks
#     })


def get_weather_data(city_name, start_date, end_date, api_key):
    # Geocodifica la città in lat/lon (puoi sostituire con un sistema più robusto)
    # geocoding_url = f"https://nominatim.openstreetmap.org/search?format=json&q={city_name}"
    # geocode_resp = requests.get(geocoding_url).json()

    geocoding_url = f"https://nominatim.openstreetmap.org/search?format=json&q={city_name}"

    geocode_response = requests.get(geocoding_url, headers={"User-Agent": "Mozilla/5.0"})
    #st.write("Geocoding response:", geocode_response.text)

    try:
        geocode_resp = geocode_response.json()
    except ValueError:
        st.error(
            "❌ Errore nel parsing della risposta JSON dal geocoding. Controlla il nome della città o la connessione.")
        st.stop()

    if not geocode_resp:
        st.error("❌ Città non trovata. Controlla che il nome sia corretto.")
        st.stop()

    if not geocode_resp:
        raise ValueError("Città non trovata.")

    lat = geocode_resp[0]['lat']
    lon = geocode_resp[0]['lon']

    # Tomorrow.io: richiedi previsioni orarie
    url = "https://api.tomorrow.io/v4/weather/forecast"
    params = {
        "location": f"{lat},{lon}",
        "apikey": api_key,
        "timesteps": "1h",  # previsione oraria
        "units": "metric",
        "fields": "temperature,humidity"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if "timelines" not in data or "hourly" not in data["timelines"]:
        raise ValueError("Risposta inattesa dall'API di Tomorrow.io")

    temps = []
    humidities = []
    dates = []
    hours = []
    dayofweeks = []

    for item in data["timelines"]["hourly"]:
        dt = datetime.datetime.fromisoformat(item["time"].replace("Z", "+00:00"))

        if start_date <= dt.date() <= end_date:
            values = item["values"]
            temps.append(values.get("temperature"))
            humidities.append(values.get("humidity"))
            dates.append(dt.date())
            hours.append(dt.hour)
            dayofweeks.append(dt.weekday())

    return pd.DataFrame({
        'Date': dates,
        'Temperature': temps,
        'Humidity': humidities,
        'Hour': hours,
        'DayOfWeek': dayofweeks
    })


def prepare_features(weather_df):
    df = weather_df.copy()

    df['day_of_week'] = df['DayOfWeek']
    df['hour'] = df['Hour']
    df['month'] = pd.to_datetime(df['Date']).dt.month
    df['day_of_year'] = pd.to_datetime(df['Date']).dt.dayofyear
    df['week_of_year'] = pd.to_datetime(df['Date']).dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    return df[['Temperature', 'Humidity', 'hour', 'day_of_week', 'month',
               'day_of_year', 'week_of_year', 'is_weekend']]

def predict_from_api_page():
    st.title("Predizione Consumo Energetico da Meteo Città (Giornaliera)")

    #api_key = "3001a620158131c941d011ba003258b3"
    api_key = "rzwVU2IQ4Lhvk5TBEpfrMAMuMTgOPDYT"
    city_name = st.text_input("Inserisci il nome della città", value="Lecce")

    today = datetime.date.today()
    start_date = st.date_input("Data inizio", value=today)
    end_date = st.date_input("Data fine", value=today + datetime.timedelta(days=2))

    model_path = "rf_energy_model.joblib"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error("Modello non trovato. Assicurati di aver salvato 'rf_energy_model.joblib'.")
        return

    if st.button("Predici Consumo Energetico Giornaliero"):
        if start_date > end_date:
            st.error("La data di inizio deve essere precedente a quella di fine.")
        else:
            try:
                weather_df = get_weather_data(city_name, start_date, end_date, api_key)
                if weather_df.empty:
                    st.error("Nessun dato meteo disponibile per il periodo richiesto.")
                    return

                # === Aggregazione per giorno ===
                daily_df = weather_df.groupby('Date').agg({
                    'Temperature': ['mean', 'min', 'max'],
                    'Humidity': ['mean', 'min', 'max']
                })
                daily_df.columns = ['temp_mean', 'temp_min', 'temp_max', 'hum_mean', 'hum_min', 'hum_max']
                daily_df.reset_index(inplace=True)

                # Aggiunta delle feature temporali
                daily_df['day_of_week'] = pd.to_datetime(daily_df['Date']).dt.dayofweek
                daily_df['is_weekend'] = daily_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
                daily_df['month'] = pd.to_datetime(daily_df['Date']).dt.month
                daily_df['week_of_year'] = pd.to_datetime(daily_df['Date']).dt.isocalendar().week

                # Predizione
                input_features = daily_df[['temp_mean', 'temp_min', 'temp_max',
                                           'hum_mean', 'hum_min', 'hum_max',
                                           'day_of_week', 'is_weekend', 'month', 'week_of_year']]

                preds = model.predict(input_features)
                daily_df['PredictedEnergyConsumption'] = preds

                # Output
                st.subheader("Risultati Predizione Giornaliera")
                st.dataframe(daily_df[['Date', 'PredictedEnergyConsumption']])

                # Grafico
                plt.figure(figsize=(10, 4))
                plt.plot(daily_df['Date'], daily_df['PredictedEnergyConsumption'], marker='o')

                # Imposta i tick esattamente sulle date dei punti
                plt.xticks(daily_df['Date'], [d.strftime('%Y-%m-%d') for d in daily_df['Date']], rotation=0)

                # Aggiunta etichette su ogni punto
                for x, y in zip(daily_df['Date'], daily_df['PredictedEnergyConsumption']):
                    plt.text(x + pd.Timedelta(hours=6), y + 0.2, f"{y:.1f}", ha='left', va='bottom', fontsize=9,
                             color='blue')

                plt.title(f"Consumo Energetico Giornaliero Previsto per {city_name}")
                plt.xlabel("Data")
                plt.ylabel("Consumo Energetico (kWh)")
                plt.grid(True)
                st.pyplot(plt)


            except Exception as e:
                st.error(f"Errore durante la predizione: {e}")

# def predict_from_api_page():
#     st.title("Predizione Consumo Energetico da Meteo Città")
#
#     api_key = "3001a620158131c941d011ba003258b3"
#     city_name = st.text_input("Inserisci il nome della città", value="Milano")
#
#     today = datetime.date.today()
#     start_date = st.date_input("Data inizio", value=today)
#     end_date = st.date_input("Data fine", value=today + datetime.timedelta(days=2))
#
#     model_path = "rf_energy_model.joblib"
#     try:
#         model = joblib.load(model_path)
#     except FileNotFoundError:
#         st.error("Modello non trovato. Assicurati di aver salvato 'rf_energy_model.joblib'.")
#         return
#
#     if st.button("Predici Consumo Energetico"):
#         if start_date > end_date:
#             st.error("La data di inizio deve essere precedente a quella di fine.")
#         else:
#             try:
#                 weather_df = get_weather_data(city_name, start_date, end_date, api_key)
#                 if weather_df.empty:
#                     st.error("Nessun dato meteo disponibile per il periodo richiesto.")
#                 else:
#                     X = prepare_features(weather_df)
#                     preds = model.predict(X)
#
#                     results = weather_df[['Date', 'Hour', 'Temperature', 'Humidity']].copy()
#                     results['PredictedEnergyConsumption'] = preds
#
#                     st.subheader("Risultati Predizione")
#                     st.dataframe(results)
#
#                     plt.figure(figsize=(12, 5))
#                     datetimes = pd.to_datetime(results['Date'].astype(str)) + pd.to_timedelta(results['Hour'], unit='h')
#                     plt.plot(datetimes, results['PredictedEnergyConsumption'], marker='o')
#                     plt.xlabel('Data e Ora')
#                     plt.ylabel('Consumo Energetico Predetto (kWh)')
#                     plt.title(f"Consumo Energetico Previsto ogni 3 ore per {city_name}")
#                     plt.xticks(rotation=45)
#                     plt.grid(True)
#                     st.pyplot(plt)
#
#             except Exception as e:
#                 st.error(f"Errore durante la predizione: {e}")


# import streamlit as st
# import requests
# import datetime
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
#
# def get_weather_data(city_name, start_date, end_date, api_key):
#     url = f"https://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={api_key}&units=metric"
#     response = requests.get(url)
#     response.raise_for_status()
#     data = response.json()
#
#     temps = []
#     humidities = []
#     dates = []
#     hours = []
#     dayofweeks = []
#
#     for item in data.get('list', []):
#         dt_txt = item['dt_txt']
#         dt = datetime.datetime.strptime(dt_txt, "%Y-%m-%d %H:%M:%S")
#
#         if start_date <= dt.date() <= end_date:
#             temps.append(item['main']['temp'])
#             humidities.append(item['main']['humidity'])
#             dates.append(dt.date())
#             hours.append(dt.hour)
#             dayofweeks.append(dt.weekday())  # 0=Lunedì, 6=Domenica
#
#     return pd.DataFrame({
#         'Date': dates,
#         'Temperature': temps,
#         'Humidity': humidities,
#         'Hour': hours,
#         'DayOfWeek': dayofweeks
#     })
#
#
# def prepare_features(weather_df):
#     return weather_df[['Temperature', 'Humidity', 'Hour', 'DayOfWeek']]
#
#
# def predict_from_api_page():
#     st.title("Predizione Consumo Energetico da Meteo Città")
#
#     # api_key = st.text_input("Inserisci la tua API key OpenWeatherMap:", type="password")
#     api_key = "3001a620158131c941d011ba003258b3"
#     city_name = st.text_input("Inserisci il nome della città", value="Milano")
#
#     today = datetime.date.today()
#     start_date = st.date_input("Data inizio", value=today)
#     end_date = st.date_input("Data fine", value=today + datetime.timedelta(days=2))
#
#     model_path = "rf_energy_model.joblib"
#     model = joblib.load(model_path)
#
#     if st.button("Predici Consumo Energetico"):
#         if start_date > end_date:
#             st.error("La data di inizio deve essere precedente a quella di fine.")
#         else:
#             try:
#                 weather_df = get_weather_data(city_name, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), api_key)
#                 if weather_df.empty:
#                     st.error("Nessun dato meteo disponibile per il periodo richiesto.")
#                 else:
#                     # Media giornaliera per evitare ripetizioni ogni 3 ore
#                     weather_df = weather_df.groupby("Date").mean().reset_index()
#
#                     X = prepare_features(weather_df)
#                     preds = model.predict(X)
#
#                     results = weather_df[['Date']].copy()
#                     results['PredictedEnergyConsumption'] = preds
#
#                     st.subheader("Risultati Predizione")
#                     st.dataframe(results)
#
#                     plt.figure(figsize=(10, 5))
#                     plt.plot(pd.to_datetime(results['Date']), results['PredictedEnergyConsumption'], marker='o')
#                     plt.xlabel('Data')
#                     plt.ylabel('Consumo Energetico Predetto (kWh)')
#                     plt.title(f"Consumo Energetico Previsto per {city_name}")
#                     plt.grid(True)
#                     st.pyplot(plt)
#
#             except Exception as e:
#                 st.error(f"Errore durante la predizione: {e}")
