import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Importa le pagine esterne
from data_preprocessing import clean_dataset
from forecast_dashboard import forecast_dashboard_page
from dati_home import home_data_page
from streamlit_historical_data import historical_data_page
from predict_from_api import predict_from_api_page



# --- Pulizia e caricamento dataset ---
clean_dataset()
data = pd.read_csv('file_ripulito.csv')

# --- Feature engineering ---
data['Weekend'] = data['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
numeric_features = ['Temperature', 'Humidity', 'hour', 'day_of_week', 'month', 'day_of_year', 'week_of_year', 'is_weekend']
target = 'EnergyConsumption'

# --- Carica modello e dati test ---
model = None
X_test = None
try:
    model = joblib.load("rf_energy_model.joblib")
except FileNotFoundError:
    st.error("Modello non trovato. Assicurati che 'rf_energy_model.joblib' sia stato creato.")
    st.stop()

# Carica dati test
try:
    X_test, y_test = joblib.load("test_data.joblib")
except FileNotFoundError:
    st.error("Dati di test non trovati. Assicurati che 'test_data.joblib' sia presente.")
    st.stop()

#
# X_test, y_test = joblib.load("rf_energy_model.joblib")
#
# if joblib.load("rf_energy_model.joblib") is None:
#     st.error("Modello non trovato. Assicurati che 'rf_energy_model.joblib' sia stato creato.")
#     st.stop()

X = data[numeric_features]
y = data[target]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


y_pred = model.predict(X_test)


# Sidebar menu
st.sidebar.title("Menu")
page = st.sidebar.radio("Vai a:", ["Analisi Dati", "Predizione Manuale", "Predizione da Meteo API",
                                   "Dashboard Previsioni",
                                   "Dati Abitazione"])

if model is None and page in ["Analisi Dati", "Predizione Manuale"]:
    st.error("Modello non trovato. Assicurati che 'rf_energy_model.joblib' sia stato creato.")
    st.stop()

# === Pagina: Analisi Dati ===
if page == "Analisi Dati":
    st.title("Analisi Esplorativa dei Dati")

    # Boxplot Weekday vs Weekend
    st.subheader("Consumo Energetico: Weekday vs Weekend")
    fig1, ax1 = plt.subplots()
    sns.boxplot(x='Weekend', y='EnergyConsumption', data=data, ax=ax1)
    st.pyplot(fig1)

    # Scatter Temperature vs Energy Consumption
    st.subheader("Relazione tra Temperatura e Consumo Energetico")
    fig2, ax2 = plt.subplots()
    # sns.scatterplot(x='Temperature', y='EnergyConsumption', data=data, ax=ax2)
    sns.scatterplot(x='Temperature', y='EnergyConsumption', data=data, ax=ax2, alpha=0.3)
    sns.regplot(x='Temperature', y='EnergyConsumption', data=data, ax=ax2, scatter=False, color='red')
    st.pyplot(fig2)

    # Umidità vs Consumo Energetico
    st.subheader("Relazione tra Umidità e Consumo Energetico")
    fig3, ax3 = plt.subplots()
    # sns.scatterplot(x="Humidity", y="EnergyConsumption", data=data, ax=ax3)
    sns.scatterplot(x="Humidity", y="EnergyConsumption", data=data, ax=ax3, alpha=0.3)
    sns.regplot(x='Humidity', y='EnergyConsumption', data=data, ax=ax3, scatter=False, color='red')
    ax3.set_title("Umidità vs Consumo Energetico")
    st.pyplot(fig3)

    # Predizione sullo stesso dataset
    y_pred = model.predict(X_test)

    # Importanza delle variabili
    st.subheader("Importanza delle Variabili")
    importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # importance = pd.DataFrame({
    #     'Feature': model.feature_names_in_,
    #     'Importance': model.feature_importances_
    # }).sort_values(by='Importance', ascending=False)
    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=importance, ax=ax_imp)
    st.pyplot(fig_imp)

    # Valori Reali vs Predetti
    st.subheader("Valori Reali vs Predetti")
    fig4, ax4 = plt.subplots()
    # sns.scatterplot(x=y, y=y_pred, ax=ax4)
    sns.scatterplot(x=y_test, y=y_pred, ax=ax4)
    ax4.set_xlabel("Reale")
    ax4.set_ylabel("Predetto")
    st.pyplot(fig4)

    # Metriche
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown("###Metriche del Modello")
    rmse = mse ** 0.5
    st.write(f"**R² (R-squared):** {r2:.4f}")
    st.write(f"**MSE (Mean Squared Error):** {mse:.2f}")
    st.write(f"**RMSE (Mean Squared Error):** {rmse:.2f}")

    # Distribuzione Reale vs Predetto
    st.subheader("Distribuzione: Reale vs Predetto")
    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
    # sns.histplot(y, label='Reale', kde=True, ax=ax_dist)
    # sns.histplot(y_pred, label='Predetto', kde=True, ax=ax_dist)
    sns.histplot(y_test, label='Reale', kde=True, ax=ax_dist)
    sns.histplot(y_pred, label='Predetto', kde=True, ax=ax_dist)
    ax_dist.legend()
    st.pyplot(fig_dist)



    # Andamento orario
    st.subheader("Consumo Energetico nelle Ore del Giorno")
    fig_hour, ax_hour = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='hour', y='EnergyConsumption', data=data, ax=ax_hour, ci=None)
    ax_hour.set_title("Consumo Energetico vs Ora")
    st.pyplot(fig_hour)

    # Confronto linea Reale vs Predetto
    st.subheader("Confronto: Reale vs Predetto")
    fig_cmp, ax_cmp = plt.subplots(figsize=(12, 6))
    # ax_cmp.plot(y.values[:100], label='Reale', marker='o')  # Limita a 100 per performance
    # ax_cmp.plot(y_pred[:100], label='Predetto', marker='x')
    ax_cmp.plot(y_test.values[:100], label='Reale', marker='o')
    ax_cmp.plot(y_pred[:100], label='Predetto', marker='x')
    ax_cmp.set_xlabel("Indice")
    ax_cmp.set_ylabel("Consumo Energetico")
    ax_cmp.legend()
    st.pyplot(fig_cmp)

    # Errore assoluto (opzionale)
    N = min(len(y_test), len(y_pred), 100)
    error = abs(y_test.values[:N] - y_pred[:N])
    fig_err, ax_err = plt.subplots(figsize=(12, 3))
    ax_err.bar(range(N), error, color='gray')
    ax_err.set_title("Errore Assoluto (|Reale - Predetto|)")
    st.pyplot(fig_err)


# === Pagina: Predizione Manuale ===
elif page == "Predizione Manuale":
    st.title("Predizione Consumo Energetico Manuale")

    st.subheader("Inserisci i dati medi giornalieri per la previsione:")

    # Inserimento dei valori medi giornalieri
    temp_mean = st.number_input("Temperatura media (°C)", value=22.0)
    temp_min = st.number_input("Temperatura minima (°C)", value=18.0)
    temp_max = st.number_input("Temperatura massima (°C)", value=26.0)

    hum_mean = st.number_input("Umidità media (%)", value=50.0)
    hum_min = st.number_input("Umidità minima (%)", value=40.0)
    hum_max = st.number_input("Umidità massima (%)", value=60.0)

    selected_date = st.date_input("Data per la previsione")

    #  Calcolo delle feature temporali dalla data
    selected_date = pd.to_datetime(selected_date)
    day_of_week = selected_date.dayofweek
    is_weekend = int(day_of_week in [5, 6])
    month = selected_date.month
    week_of_year = selected_date.isocalendar().week

    # Costruzione dataframe di input
    input_data = pd.DataFrame([[
        temp_mean, temp_min, temp_max,
        hum_mean, hum_min, hum_max,
        day_of_week, is_weekend,
        month, week_of_year
    ]], columns=[
        'temp_mean', 'temp_min', 'temp_max',
        'hum_mean', 'hum_min', 'hum_max',
        'day_of_week', 'is_weekend',
        'month', 'week_of_year'
    ])

    if st.button("Predici Consumo Giornaliero"):
        prediction = model.predict(input_data)
        st.success(f"📅 Consumo energetico previsto per il {selected_date.date()}: **{prediction[0]:.2f} kWh**")
#####################################################################################################################

# === Altre Pagine ===
elif page == "Predizione da Meteo API":
    predict_from_api_page()

elif page == "Dashboard Previsioni":
    forecast_dashboard_page()

elif page == "Dati Abitazione":
    home_data_page()



###########################################################################################

# import streamlit as st
# import joblib
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
#
# # Importa le pagine esterne
# from data_preprocessing import clean_dataset
# from realtime_data import realtime_data_page
# from streamlit_historical_data import historical_data_page
# from predict_from_api import predict_from_api_page
#
# # --- Caricamento dataset ---
#
# clean_dataset()
#
# data = pd.read_csv('file_ripulito.csv')
#
# # Definizione delle feature e del target
# numeric_features = ['Temperature', 'Humidity', 'hour', 'day_of_week']
# target = 'EnergyConsumption'
#
# data['Weekend'] = data['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
#
#
# X = data[numeric_features]
# y = data[target]
#
# # Sidebar menu
# st.sidebar.title("Menu")
# page = st.sidebar.radio("Vai a:", ["Analisi Dati", "Predizione Manuale", "Predizione da Meteo API",
#                                    "Dati Storici", "Dati Istantanei"])
#
# if page == "Analisi Dati":
#     st.title("Analisi Esplorativa dei Dati")
#
#     # Boxplot Weekday vs Weekend
#     st.subheader("Consumo Energetico: Weekday vs Weekend")
#     fig1, ax1 = plt.subplots()
#     sns.boxplot(x='Weekend', y='EnergyConsumption', data=data, ax=ax1)
#     st.pyplot(fig1)
#
#     # Scatter Temperature vs Energy Consumption
#     st.subheader("Relazione tra Temperatura e Consumo Energetico")
#     fig2, ax2 = plt.subplots()
#     sns.scatterplot(x='Temperature', y='EnergyConsumption', data=data, ax=ax2)
#     st.pyplot(fig2)
#
#     # 3. Scatterplot: Umidità vs Consumo Energetico
#     st.subheader("Relazione tra Umidità e Consumo Energetico")
#     fig3, ax3 = plt.subplots()
#     sns.scatterplot(x="Humidity", y="EnergyConsumption", data=data, ax=ax3)
#     ax3.set_title("Umidità vs Consumo Energetico")
#     st.pyplot(fig3)
#
#     # Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
#
#     # Modello Random Forest
#     model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     # Importanza delle variabili
#     st.subheader("Importanza delle variabili")
#     importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
#     fig3, ax3 = plt.subplots(figsize=(8,6))
#     sns.barplot(x='Importance', y='Feature', data=importance, ax=ax3)
#     st.pyplot(fig3)
#
#     # Valori Predetti vs Reali
#     st.subheader("Valori Predetti vs Reali")
#     fig4, ax4 = plt.subplots()
#     sns.scatterplot(x=y_test, y=y_pred, ax=ax4)
#     ax4.set_xlabel("Valori Reali")
#     ax4.set_ylabel("Valori Predetti")
#     st.pyplot(fig4)
#
#     # Metriche
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
#     st.write(f"**R-squared (R²):** {r2:.4f}")
#
#     # Distribuzione del Consumo Energetico: Reale vs Predetto
#     st.subheader("Distribuzione del Consumo Energetico: Reale vs Predetto")
#     fig_dist, ax_dist = plt.subplots(figsize=(10,6))
#     sns.histplot(y_test, label='Reale', kde=True, ax=ax_dist)
#     sns.histplot(y_pred, label='Predetto', kde=True, ax=ax_dist)
#     ax_dist.legend()
#     st.pyplot(fig_dist)
#
#     st.subheader("Andamento del Consumo Energetico nelle Ore del Giorno")
#     fig4, ax4 = plt.subplots(figsize=(12, 6))
#     sns.lineplot(x='hour', y='EnergyConsumption', data=data, ax=ax4, ci=None)
#     ax4.set_title('Consumo Energetico in funzione dell’Ora del Giorno')
#     ax4.set_xlabel('Ora del Giorno')
#     ax4.set_ylabel('Consumo Energetico')
#     st.pyplot(fig4)
#
#
#     # 2. Grafico Reale vs Predetto (Assume che y_test e y_pred siano già disponibili)
#     st.subheader("Confronto: Valori Reali vs Predetti")
#     fig_comparison, ax_comparison = plt.subplots(figsize=(12, 6))
#     ax_comparison.plot(y_test.values, label='Reale', marker='o')
#     ax_comparison.plot(y_pred, label='Predetto', marker='x')
#     ax_comparison.set_xlabel('Indice')
#     ax_comparison.set_ylabel('Consumo Energetico')
#     ax_comparison.set_title('Valori Reali vs. Valori Predetti')
#     ax_comparison.legend()
#     st.pyplot(fig_comparison)
#
#
# elif page == "Predizione Manuale":
#     st.title("Predizione Consumo Energetico Manuale")
#
#     # Carica modello addestrato
#     try:
#         model = joblib.load("rf_energy_model.joblib")
#         feature_names = model.feature_names_in_
#         st.write("Feature attese dal modello:", feature_names)
#     except FileNotFoundError:
#         st.error("Modello non trovato. Assicurati di aver salvato 'rf_energy_model.joblib'.")
#     else:
#         st.subheader("Inserisci i dati per la previsione:")
#
#         temperature = st.number_input("Temperatura (°C)", value=22.0)
#         humidity = st.number_input("Umidità (%)", value=50.0)
#         hour = st.number_input("Ora del giorno (0-23)", min_value=0, max_value=23, value=12)
#         dayofweek = st.selectbox("Giorno della settimana (0=Lun - 6=Dom)", list(range(7)))
#
#         # Costruzione feature derivate da una data fittizia
#         base_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(dayofweek))  # giorno della settimana
#
#         day_of_year = base_date.dayofyear
#         month = base_date.month
#         week_of_year = base_date.isocalendar().week
#         is_weekend = int(dayofweek >= 5)
#
#         # Costruzione dell'input coerente con il modello
#         input_data = pd.DataFrame([[temperature, humidity, hour, dayofweek,
#                                     month, day_of_year, week_of_year, is_weekend]],
#                                   columns=['Temperature', 'Humidity', 'hour', 'day_of_week',
#                                            'month', 'day_of_year', 'week_of_year', 'is_weekend'])
#
#         # Riorganizza l'ordine delle colonne in base a quello del modello
#         input_data = input_data[feature_names]
#
#         if st.button("Predici Consumo Energetico"):
#             prediction = model.predict(input_data)
#             st.success(f"Consumo Energetico Predetto: {prediction[0]:.2f} kWh")
#
# elif page == "Predizione da Meteo API":
#     predict_from_api_page()  # funzione dal modulo esterno
#
# elif page == "Dati Storici":
#     historical_data_page()  # funzione dal modulo esterno
#
# elif page == "Dati Istantanei":
#     realtime_data_page()  # funzione dal modulo esterno







# import streamlit as st
# import joblib
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
#
# from realtime_data import realtime_data_page
# from streamlit_historical_data import historical_data_page
# from predict_from_api import predict_from_api_page  # nuova pagina predizione API
#
# # Carica il dataset
# data = pd.read_csv('file_ripulito.csv')
#
# # Definisci features (X) e target (y)
# # X = pd[['Temperature', 'Humidity']]
# # y = pd['EnergyConsumption']
#
# data['Date'] = pd.to_datetime(data['Date'])
#
# # Estrai giorno della settimana (0=lunedì, 6=domenica)
# data['DayOfWeek'] = data['Date'].dt.dayofweek
#
# # Variabili usate
# numeric_features = ['Temperature', 'Humidity', 'Hour', 'DayOfWeek']
# target = 'EnergyConsumption'
#
# X = data[numeric_features]
# y = data[target]
#
# # Sidebar menu
# st.sidebar.title("Menu")
# page = st.sidebar.radio("Vai a:", ["Analisi Dati", "Predizione Manuale", "Predizione da Meteo API",
#                                    "Dati Storici", "Dati Istantanei"])
#
# if page == "Analisi Dati":
#     st.title("Analisi Esplorativa dei Dati")
#
#     # Boxplot Weekday vs Weekend
#     st.subheader("Consumo Energetico: Weekday vs Weekend")
#     fig1, ax1 = plt.subplots()
#     sns.boxplot(x='Weekend', y='EnergyConsumption', data=data, ax=ax1)
#     st.pyplot(fig1)
#
#     # Histogram Temperatura vs Consumo
#     st.subheader("Relazione tra Temperatura e Consumo Energetico")
#     fig2, ax2 = plt.subplots()
#     sns.scatterplot(x="Temperature", y="EnergyConsumption", data=data, ax=ax2)
#     st.pyplot(fig2)
#
#     # Suddivisione train/test
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Modello Random Forest
#     model = RandomForestRegressor(random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#
#     # Importanza delle variabili
#     st.subheader("Importanza delle variabili")
#     importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
#     importance = importance.sort_values(by="Importance", ascending=False)
#
#     fig3, ax3 = plt.subplots(figsize=(8, 6))
#     sns.barplot(x="Importance", y="Feature", data=importance, ax=ax3)
#     st.pyplot(fig3)
#
#     # Valori Predetti vs Reali
#     st.subheader("Valori Predetti vs Reali")
#     fig4, ax4 = plt.subplots()
#     sns.scatterplot(x=y_test, y=y_pred, ax=ax4)
#     ax4.set_xlabel("Valori Reali")
#     ax4.set_ylabel("Valori Predetti")
#     st.pyplot(fig4)
#
#     # Metriche
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
#     st.write(f"**R-squared (R²):** {r2:.4f}")
#
#     # Distribuzione del Consumo Energetico: Reale vs Predetto
#     st.subheader("Distribuzione del Consumo Energetico: Reale vs Predetto")
#     fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
#     sns.histplot(y_test, label='Reale', kde=True, ax=ax_dist)
#     sns.histplot(y_pred, label='Predetto', kde=True, ax=ax_dist)
#     ax_dist.set_xlabel('Consumo Energetico')
#     ax_dist.set_ylabel('Frequenza')
#     ax_dist.set_title('Distribuzione del Consumo Energetico: Reale vs. Predetto')
#     ax_dist.legend()
#     st.pyplot(fig_dist)
#
#     # Grafico Reale vs Predetto
#     fig_comparison, ax_comparison = plt.subplots(figsize=(12, 6))
#     ax_comparison.plot(y_test.values, label='Reale', marker='o')
#     ax_comparison.plot(y_pred, label='Predetto', marker='x')
#     ax_comparison.set_xlabel('Indice')
#     ax_comparison.set_ylabel('Consumo Energetico')
#     ax_comparison.set_title('Valori Reali vs. Valori Predetti')
#     ax_comparison.legend()
#     st.pyplot(fig_comparison)
#
#     #
#     # # Histogram Umidità vs Consumo
#     # st.subheader("Relazione tra Umidità e Consumo Energetico")
#     # fig3, ax3 = plt.subplots()
#     # sns.scatterplot(x="Humidity", y="EnergyConsumption", data=data, ax=ax3)
#     # st.pyplot(fig3)
#     #
#     # # Line Plot: Energy Consumption vs Hour of Day
#     # st.subheader("Andamento del Consumo Energetico nelle Ore del Giorno")
#     # fig4, ax4 = plt.subplots(figsize=(12, 6))
#     # sns.lineplot(x='Hour', y='EnergyConsumption', data=data, ax=ax4, ci=None)
#     # ax4.set_title('Consumo Energetico in funzione dell’Ora del Giorno')
#     # ax4.set_xlabel('Ora del Giorno')
#     # ax4.set_ylabel('Consumo Energetico')
#     # st.pyplot(fig4)
#     #
#     # # Random Forest su subset per vedere feature importance
#     # from sklearn.ensemble import RandomForestRegressor
#     # from sklearn.model_selection import train_test_split
#     #
#     # numeric_features = ['Temperature', 'Humidity']
#     # categorical_features = ['HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday']
#     # target = 'EnergyConsumption'
#     #
#     # X = data[numeric_features + categorical_features]
#     # X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
#     # y = data[target]
#     #
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     #
#     # model = RandomForestRegressor(random_state=42)
#     # model.fit(X_train, y_train)
#     # y_pred = model.predict(X_test)
#     #
#     # st.subheader("Importanza delle variabili")
#     # importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
#     # importance = importance.sort_values(by="Importance", ascending=False)
#     #
#     # fig3, ax3 = plt.subplots(figsize=(8, 6))
#     # sns.barplot(x="Importance", y="Feature", data=importance, ax=ax3)
#     # st.pyplot(fig3)
#     #
#     # st.subheader("Valori Predetti vs Reali")
#     # fig4, ax4 = plt.subplots()
#     # sns.scatterplot(x=y_test, y=y_pred, ax=ax4)
#     # ax4.set_xlabel("Valori Reali")
#     # ax4.set_ylabel("Valori Predetti")
#     # st.pyplot(fig4)
#     #
#     # mse = mean_squared_error(y_test, y_pred)
#     # r2 = r2_score(y_test, y_pred)
#     # st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
#     # st.write(f"**R-squared (R²):** {r2:.4f}")
#     #
#     # # Distribuzione dei valori reali vs. predetti
#     # st.subheader("Distribuzione del Consumo Energetico: Reale vs Predetto")
#     #
#     # fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
#     # sns.histplot(y_test, label='Reale', kde=True, ax=ax_dist)
#     # sns.histplot(y_pred, label='Predetto', kde=True, ax=ax_dist)
#     # ax_dist.set_xlabel('Consumo Energetico')
#     # ax_dist.set_ylabel('Frequenza')
#     # ax_dist.set_title('Distribuzione del Consumo Energetico: Reale vs. Predetto')
#     # ax_dist.legend()
#     #
#     # st.pyplot(fig_dist)
#     #
#     # # Aggiunta del grafico Reale vs Predetto
#     # fig_comparison, ax_comparison = plt.subplots(figsize=(12, 6))
#     # ax_comparison.plot(y_test.values, label='Reale', marker='o')
#     # ax_comparison.plot(y_pred, label='Predetto', marker='x')
#     # ax_comparison.set_xlabel('Indice')
#     # ax_comparison.set_ylabel('Consumo Energetico')
#     # ax_comparison.set_title('Valori Reali vs. Valori Predetti')
#     # ax_comparison.legend()
#     #
#     # st.pyplot(fig_comparison)
#
# elif page == "Predizione Manuale":
#     st.title("Predizione Consumo Energetico Manuale")
#
#     # Carica modello già addestrato
#     model = joblib.load("rf_energy_model.joblib")
#
#     st.subheader("Inserisci i dati per la previsione:")
#
#     temperature = st.number_input("Temperatura (°C)", value=22.0)
#     humidity = st.number_input("Umidità (%)", value=50.0)
#     sqft = st.number_input("Superficie (m²)", value=1000)
#     occupancy = st.number_input("Numero occupanti", value=10)
#     renewable = st.number_input("Energia Rinnovabile Prodotta", value=200.0)
#
#     hvac = st.selectbox("Uso HVAC", ['On', 'Off'])
#     lighting = st.selectbox("Uso Luci", ['On', 'Off'])
#     dayofweek = st.selectbox("Giorno della settimana (0=Lun - 6=Dom)", list(range(7)))
#     holiday = st.selectbox("È un giorno festivo?", ['Yes', 'No'])
#
#     input_data = {
#         'Temperature': temperature,
#         'Humidity': humidity,
#         'SquareFootage': sqft,
#         'Occupancy': occupancy,
#         'RenewableEnergy': renewable,
#         'HVACUsage_On': 1 if hvac == 'On' else 0,
#         'LightingUsage_On': 1 if lighting == 'On' else 0,
#         'DayOfWeek_1': 0, 'DayOfWeek_2': 0, 'DayOfWeek_3': 0,
#         'DayOfWeek_4': 0, 'DayOfWeek_5': 0, 'DayOfWeek_6': 0,
#         'Holiday_Yes': 1 if holiday == 'Yes' else 0
#     }
#     if f'DayOfWeek_{dayofweek}' in input_data:
#         input_data[f'DayOfWeek_{dayofweek}'] = 1
#
#     input_df = pd.DataFrame([input_data])
#
#     if st.button("Predici Consumo Energetico"):
#         prediction = model.predict(input_df)[0]
#         st.success(f"Consumo Energetico Previsto: **{prediction:.2f} kWh**")
#
# elif page == "Predizione da Meteo API":
#     predict_from_api_page()  # richiama la funzione del modulo esterno
# elif page == "Dati Storici":
#     historical_data_page()
# elif page == "Dati Istantanei":
#     realtime_data_page()
#
#
#
# # import pandas as pd
# # import numpy as np
# # import joblib
# # import streamlit as st
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # from sklearn.metrics import mean_squared_error, r2_score
# #
# # # Carica il dataset
# # data = pd.read_csv("C:/Users/giuse/Downloads/Energy_consumption.csv")
# #
# # # Pre-elaborazione
# # data['Timestamp'] = pd.to_datetime(data['Timestamp'])
# # data['Hour'] = data['Timestamp'].dt.hour
# # data['Day'] = data['Timestamp'].dt.day
# # data['Month'] = data['Timestamp'].dt.month
# # data['Year'] = data['Timestamp'].dt.year
# # data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
# # data['Weekend'] = data['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
# #
# # # Sidebar
# # st.sidebar.title("Menu")
# # page = st.sidebar.radio("Vai a:", ["📊 Analisi Dati", "🤖 Predizione Manuale", "🌤️ Predizione da Meteo API"])
# #
# # # === SEZIONE ANALISI DATI ===
# # if page == "📊 Analisi Dati":
# #     st.title("Analisi Esplorativa dei Dati")
# #
# #     # Boxplot Weekday vs Weekend
# #     st.subheader("Consumo Energetico: Weekday vs Weekend")
# #     fig1, ax1 = plt.subplots()
# #     sns.boxplot(x='Weekend', y='EnergyConsumption', data=data, ax=ax1)
# #     st.pyplot(fig1)
# #
# #     # Histogram Temperatura vs Consumo
# #     st.subheader("Relazione tra Temperatura e Consumo Energetico")
# #     fig2, ax2 = plt.subplots()
# #     sns.scatterplot(x="Temperature", y="EnergyConsumption", data=data, ax=ax2)
# #     st.pyplot(fig2)
# #
# #     # Histogram Umidità vs Consumo
# #     st.subheader("Relazione tra Umidità e Consumo Energetico")
# #     fig3, ax3 = plt.subplots()
# #     sns.scatterplot(x="Humidity", y="EnergyConsumption", data=data, ax=ax3)
# #     st.pyplot(fig3)
# #
# #     # Line Plot: Energy Consumption vs Hour of Day
# #     st.subheader("Andamento del Consumo Energetico nelle Ore del Giorno")
# #     fig4, ax4 = plt.subplots(figsize=(12, 6))
# #     sns.lineplot(x='Hour', y='EnergyConsumption', data=data, ax=ax4, ci=None)
# #     ax4.set_title('Consumo Energetico in funzione dell’Ora del Giorno')
# #     ax4.set_xlabel('Ora del Giorno')
# #     ax4.set_ylabel('Consumo Energetico')
# #     st.pyplot(fig4)
# #
# #     # Random Forest su subset per vedere feature importance
# #     from sklearn.ensemble import RandomForestRegressor
# #     from sklearn.model_selection import train_test_split
# #
# #     numeric_features = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy']
# #     categorical_features = ['HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday']
# #     target = 'EnergyConsumption'
# #
# #     X = data[numeric_features + categorical_features]
# #     X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
# #     y = data[target]
# #
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #
# #     model = RandomForestRegressor(random_state=42)
# #     model.fit(X_train, y_train)
# #     y_pred = model.predict(X_test)
# #
# #     st.subheader("Importanza delle variabili")
# #     importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
# #     importance = importance.sort_values(by="Importance", ascending=False)
# #
# #     fig3, ax3 = plt.subplots(figsize=(8, 6))
# #     sns.barplot(x="Importance", y="Feature", data=importance, ax=ax3)
# #     st.pyplot(fig3)
# #
# #     st.subheader("Valori Predetti vs Reali")
# #     fig4, ax4 = plt.subplots()
# #     sns.scatterplot(x=y_test, y=y_pred, ax=ax4)
# #     ax4.set_xlabel("Valori Reali")
# #     ax4.set_ylabel("Valori Predetti")
# #     st.pyplot(fig4)
# #
# #     mse = mean_squared_error(y_test, y_pred)
# #     r2 = r2_score(y_test, y_pred)
# #     st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
# #     st.write(f"**R-squared (R²):** {r2:.4f}")
# #
# #     # Distribuzione dei valori reali vs. predetti
# #     st.subheader("Distribuzione del Consumo Energetico: Reale vs Predetto")
# #
# #     fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
# #     sns.histplot(y_test, label='Reale', kde=True, ax=ax_dist)
# #     sns.histplot(y_pred, label='Predetto', kde=True, ax=ax_dist)
# #     ax_dist.set_xlabel('Consumo Energetico')
# #     ax_dist.set_ylabel('Frequenza')
# #     ax_dist.set_title('Distribuzione del Consumo Energetico: Reale vs. Predetto')
# #     ax_dist.legend()
# #
# #     st.pyplot(fig_dist)
# #
# #     # Aggiunta del grafico Reale vs Predetto
# #     fig_comparison, ax_comparison = plt.subplots(figsize=(12, 6))
# #     ax_comparison.plot(y_test.values, label='Reale', marker='o')
# #     ax_comparison.plot(y_pred, label='Predetto', marker='x')
# #     ax_comparison.set_xlabel('Indice')
# #     ax_comparison.set_ylabel('Consumo Energetico')
# #     ax_comparison.set_title('Valori Reali vs. Valori Predetti')
# #     ax_comparison.legend()
# #
# #     st.pyplot(fig_comparison)
# #
# #
# # # === SEZIONE PREDIZIONE ===
# # elif page == "🤖 Predizione":
# #     st.title("Predizione Consumo Energetico")
# #
# #     # Carica modello già addestrato
# #     model = joblib.load("rf_energy_model.joblib")
# #
# #     st.subheader("Inserisci i dati per la previsione:")
# #
# #     temperature = st.number_input("Temperatura (°C)", value=22.0)
# #     humidity = st.number_input("Umidità (%)", value=50.0)
# #     sqft = st.number_input("Superficie (m²)", value=1000)
# #     occupancy = st.number_input("Numero occupanti", value=10)
# #     renewable = st.number_input("Energia Rinnovabile Prodotta", value=200.0)
# #
# #     hvac = st.selectbox("Uso HVAC", ['On', 'Off'])
# #     lighting = st.selectbox("Uso Luci", ['On', 'Off'])
# #     dayofweek = st.selectbox("Giorno della settimana (0=Lun - 6=Dom)", list(range(7)))
# #     holiday = st.selectbox("È un giorno festivo?", ['Yes', 'No'])
# #
# #     # Crea input per il modello
# #     input_data = {
# #         'Temperature': temperature,
# #         'Humidity': humidity,
# #         'SquareFootage': sqft,
# #         'Occupancy': occupancy,
# #         'RenewableEnergy': renewable,
# #         'HVACUsage_On': 1 if hvac == 'On' else 0,
# #         'LightingUsage_On': 1 if lighting == 'On' else 0,
# #         'DayOfWeek_1': 0, 'DayOfWeek_2': 0, 'DayOfWeek_3': 0,
# #         'DayOfWeek_4': 0, 'DayOfWeek_5': 0, 'DayOfWeek_6': 0,
# #         'Holiday_Yes': 1 if holiday == 'Yes' else 0
# #     }
# #     if f'DayOfWeek_{dayofweek}' in input_data:
# #         input_data[f'DayOfWeek_{dayofweek}'] = 1
# #
# #     input_df = pd.DataFrame([input_data])
# #
# #     # Previsione
# #     if st.button("Predici Consumo Energetico"):
# #         prediction = model.predict(input_df)[0]
# #         st.success(f"Consumo Energetico Previsto: **{prediction:.2f} kWh**")
