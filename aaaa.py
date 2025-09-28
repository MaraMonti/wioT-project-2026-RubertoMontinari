import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, UTC
import pytz
import os
import matplotlib

matplotlib.use('Agg')  # Cambia da 'TkAgg' a 'Agg'
import matplotlib.pyplot as plt

# Import per il Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # Per salvare/caricare il modello

# --- 1. CONFIGURAZIONE INIZIALE ---

# Configurazione API Home Assistant
BASE_URL = "https://gdfhome.duckdns.org/api"
# API Token fornito (Assicurati che sia sempre valido e abbia i permessi necessari!)
API_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI1MDdmNjliNjA0MmY0M2M1OTk0NjVmZDRiZmZlMWZjNyIsImlhdCI6MTcyNDc1MTMxMywiZXhwIjoyMDQwMTExMzEzfQ.S61REKbuTL1l4yP-iIQRDuZvyCmGWDJoL7-FQXAl7xg"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}

ENTITY_IDS = {
    "power": "sensor.quadro_primo_terra_channel_1_power",
    "temperature_indoor": "sensor.termostat_temperature",
    "humidity_indoor": "sensor.termostat_humidity",
}

LOCAL_TIMEZONE = pytz.timezone('Europe/Rome')

PROF_DATA_PATH = "C:/Users/giuse/Desktop/dataset/dataset_prof/consumi_01-11-2023_31_10_2024.csv"
PROF_TIME_COL = 'last_changed'  # <<< POTREBBE ESSERE NECESSARIO AGGIUSTARE QUESTO NOME se il tuo file del professore ha una colonna data/ora diversa.

PROF_ADDITIONAL_COLS = [
    'state']  # <<< Ho cambiato 'value' a 'state' basandomi sul tuo log che mostrava 'state' nel file del prof.

EXTERNAL_WEATHER_DATA_PATH = "C:/Users/Eka/Desktop/Progetto Python/API/dati_climatici/Roma_weather.csv"

KAGGLE_TIME_COL = 'DATE'  # Colonna della data nel dataset
KAGGLE_TEMP_COL = 'TAVG'  # Colonna della temperatura media nel dataset
KAGGLE_HUM_COL = None
KAGGLE_PRESSURE_COL = None
KAGGLE_WIND_SPEED_COL = None
KAGGLE_PRECIP_COL = 'PRCP'  # Colonna delle precipitazioni nel dataset


# --- 2. FUNZIONI DI ACQUISIZIONE DATI ---

def get_current_state(entity_id: str) -> dict:
    url = f"{BASE_URL}/states/{entity_id}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        data = response.json()
        if 'state' in data and 'last_changed' in data:
            return {
                'entity_id': entity_id,
                'state': data['state'],
                'last_changed': data['last_changed'],
                'last_updated': data.get('last_updated', data['last_changed'])
            }
        else:
            print(f"Avviso: Dati incompleti per '{entity_id}'. Chiavi 'state' o 'last_changed' mancanti.")
            return {}
    except requests.exceptions.RequestException as e:
        print(f"Errore API durante il recupero dello stato corrente per '{entity_id}': {e}")
        return {}


def get_historical_data(entity_id: str, start_date: datetime, end_date: datetime = None) -> list:
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    url = f"{BASE_URL}/history/period/{start_date_str}?filter_entity_id={entity_id}"


    if end_date:
        end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        url += f"&end_time={end_date_str}"
    print("URL " + url)
    print(
        f"Recupero dati storici per '{entity_id}' da {start_date.strftime('%Y-%m-%d %H:%M UTC')} a {end_date.strftime('%Y-%m-%d %H:%M UTC') if end_date else 'ora attuale UTC'}...")
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data or not isinstance(data, list) or len(data) == 0 or not isinstance(data[0], list):
            print(f"Avviso: Nessun dato storico grezzo ricevuto o formato inatteso per '{entity_id}'.")
            return []
        return data[0]
    except requests.exceptions.RequestException as e:
        print(f"Errore API durante il recupero dei dati storici per '{entity_id}': {e}")
    return []


def clean_and_process_data_to_df(raw_data: list, sensor_name: str) -> pd.DataFrame:
    if not raw_data:
        print(f"Pulizia: Nessun dato grezzo da elaborare per '{sensor_name}'.")
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)

    required_cols = ['state', 'last_changed']
    if not all(col in df.columns for col in required_cols):
        print(
            f"Errore: Colonne essenziali ({required_cols}) mancanti nel DataFrame per '{sensor_name}'. Colonne trovate: {df.columns.tolist()}")
        return pd.DataFrame()

    df['last_changed'] = pd.to_datetime(df['last_changed'], errors='coerce', utc=True)
    df.dropna(subset=['last_changed'], inplace=True)
    df = df.set_index('last_changed')

    df = df[['state']].copy()
    df.rename(columns={'state': sensor_name}, inplace=True)

    original_rows = len(df)
    df[sensor_name] = pd.to_numeric(df[sensor_name], errors='coerce')
    df.dropna(subset=[sensor_name], inplace=True)

    if len(df) < original_rows:
        print(
            f"Pulizia: Rimosse {original_rows - len(df)} righe con valori non numerici o mancanti per '{sensor_name}'.")

    if df.empty:
        print(f"Nessun dato valido rimasto per '{sensor_name}' dopo la pulizia.")
        return pd.DataFrame()

    df = df.sort_index()
    return df


def combine_all_sensor_data(dataframes: list, target_freq: str = 'h') -> pd.DataFrame:
    if not dataframes:
        print("Nessun DataFrame fornito per la combinazione.")
        return pd.DataFrame()

    # Rimuovi eventuali DataFrame vuoti dalla lista
    dataframes = [df for df in dataframes if not df.empty]
    if not dataframes:
        print("Tutti i DataFrame forniti erano vuoti dopo la pulizia. Nessun dato da combinare.")
        return pd.DataFrame()

    print(f"\nCombinazione e allineamento di {len(dataframes)} sensori alla frequenza '{target_freq}'...")

    # Resample il primo DataFrame come base
    df_combined = dataframes[0].resample(target_freq).mean()

    # Esegui un outer join per ogni DataFrame aggiuntivo dopo averlo risampled
    for i in range(1, len(dataframes)):
        df_resampled = dataframes[i].resample(target_freq).mean()
        # Aggiungi suffissi per evitare sovrapposizioni di colonne
        df_combined = df_combined.join(df_resampled, how='outer', lsuffix='_api_base', rsuffix=f'_api_sensor{i}')
        # Ho cambiato i suffissi per evitare conflitti con i nomi finali

    df_combined = df_combined.sort_index()

    # Gestione dei valori mancanti dopo il join e il resampling
    df_combined.ffill(inplace=True)
    df_combined.bfill(inplace=True)

    original_rows_combined = len(df_combined)
    df_combined.dropna(inplace=True)  # Rimuovi eventuali righe che contengono ancora NaN (es. intere colonne vuote)
    if len(df_combined) < original_rows_combined:
        print(
            f"ATTENZIONE: Rimosse {original_rows_combined - len(df_combined)} righe con NaN non gestibili dopo la combinazione/fill.")

    print(
        f"DataFrame finale combinato: {len(df_combined)} righe dal {df_combined.index.min()} al {df_combined.index.max()}")
    return df_combined


# FUNZIONE AGGIORNATA PER CARICARE PIÙ COLONNE DAL FILE DEL PROFESSORE
def load_prof_data(file_path: str, time_col: str, cols_to_load: list) -> pd.DataFrame:
    if not os.path.exists(file_path):
        print(f"Errore: File del professore non trovato: {file_path}")
        return pd.DataFrame()

    print(f"Caricamento dati del professore da: {file_path}...")
    try:
        try:
            if file_path.endswith('.xlsx'):
                df_prof = pd.read_excel(file_path)
            else:  # Assumi CSV
                df_prof = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("Tentativo di leggere il CSV del professore con encoding 'latin1'...")
            df_prof = pd.read_csv(file_path, encoding='latin1')

        # Rimuovi spazi extra dai nomi delle colonne, utile per i CSV
        df_prof.columns = df_prof.columns.str.strip()

        if time_col not in df_prof.columns:
            print(
                f"Errore: Colonna temporale '{time_col}' non trovata nel file del professore. Colonne disponibili: {df_prof.columns.tolist()}")
            return pd.DataFrame()

        # Verifica che tutte le colonne richieste siano presenti
        missing_cols = [col for col in cols_to_load if col not in df_prof.columns]
        if missing_cols:
            print(
                f"Avviso: Le seguenti colonne richieste non sono state trovate nel file del professore: {missing_cols}. Saranno ignorate.")
            # Rimuovi le colonne mancanti dalla lista da caricare per evitare errori
            cols_to_load = [col for col in cols_to_load if col not in missing_cols]
            if not cols_to_load:
                print("Nessuna colonna valida rimanente da caricare dal file del professore.")
                return pd.DataFrame()

        df_prof[time_col] = pd.to_datetime(df_prof[time_col], errors='coerce', utc=True)
        df_prof.dropna(subset=[time_col], inplace=True)
        df_prof = df_prof.set_index(time_col)

        # Seleziona solo le colonne che vuoi caricare
        df_prof = df_prof[cols_to_load].copy()

        # Rinomina la colonna 'state' se presente, per chiarezza nel DataFrame combinato (era 'value')
        if 'state' in df_prof.columns:
            df_prof.rename(columns={'state': 'prof_main_value'}, inplace=True)

        # Converte tutte le colonne del professore a numerico, gestendo gli errori
        for col in df_prof.columns:
            df_prof[col] = pd.to_numeric(df_prof[col], errors='coerce')
        df_prof.dropna(inplace=True)  # Rimuove le righe con NaN dopo la conversione

        df_prof = df_prof.sort_index()

        print(f"Dati del professore caricati: {len(df_prof)} righe dal {df_prof.index.min()} al {df_prof.index.max()}")
        return df_prof
    except Exception as e:
        print(f"Errore durante il caricamento o la pulizia dei dati del professore: {e}")
        return pd.DataFrame()


def load_external_weather_data(file_path: str,
                               time_col_name: str,
                               temp_col_name: str,
                               hum_col_name: str = None,
                               pressure_col_name: str = None,
                               wind_speed_col_name: str = None,
                               precip_col_name: str = None
                               ) -> pd.DataFrame:
    if not os.path.exists(file_path):
        print(f"Errore: File dati climatici non trovato: {file_path}")
        print(
            f"Assicurati che il file '{os.path.basename(file_path)}' sia nella cartella '{os.path.dirname(file_path)}'.")
        return pd.DataFrame()

    print(f"Caricamento dati climatici da: {file_path}...")
    try:
        try:
            if file_path.endswith('.xlsx'):
                df_weather = pd.read_excel(file_path)
            else:
                df_weather = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("Tentativo di leggere il CSV climatico con encoding 'latin1'...")
            df_weather = pd.read_csv(file_path, encoding='latin1')

        # Rimuovi eventuali spazi nelle intestazioni delle colonne del CSV di Kaggle
        df_weather.columns = df_weather.columns.str.strip()

        if time_col_name not in df_weather.columns:
            print(f"Errore: Colonna temporale '{time_col_name}' non trovata nel file climatico.")
            print(f"Colonne disponibili: {df_weather.columns.tolist()}")
            return pd.DataFrame()

        df_weather[time_col_name] = pd.to_datetime(df_weather[time_col_name], errors='coerce', utc=True)
        df_weather.dropna(subset=[time_col_name], inplace=True)
        df_weather = df_weather.set_index(time_col_name)

        weather_cols_map = {}
        if temp_col_name and temp_col_name in df_weather.columns:
            weather_cols_map[temp_col_name] = 'outdoor_temperature_c'
        if hum_col_name and hum_col_name in df_weather.columns:
            weather_cols_map[hum_col_name] = 'outdoor_humidity_perc'
        if pressure_col_name and pressure_col_name in df_weather.columns:
            weather_cols_map[pressure_col_name] = 'outdoor_pressure_mb'
        if wind_speed_col_name and wind_speed_col_name in df_weather.columns:
            weather_cols_map[wind_speed_col_name] = 'outdoor_wind_speed_kph'
        if precip_col_name and precip_col_name in df_weather.columns:
            weather_cols_map[precip_col_name] = 'outdoor_precipitation_mm'

        if not weather_cols_map:
            print("Nessuna colonna climatica pertinente riconosciuta nel file con i nomi forniti.")
            print(
                f"Colonne tentate: {time_col_name}, {temp_col_name}, {hum_col_name}, {pressure_col_name}, {wind_speed_col_name}, {precip_col_name}")
            print(f"Colonne disponibili nel file: {df_weather.columns.tolist()}")
            return pd.DataFrame()

        df_weather = df_weather[list(weather_cols_map.keys())].copy()
        df_weather.rename(columns=weather_cols_map, inplace=True)

        for col in df_weather.columns:
            df_weather[col] = pd.to_numeric(df_weather[col], errors='coerce')
        df_weather.dropna(inplace=True)

        df_weather = df_weather.sort_index()

        print(
            f"Dati climatici caricati: {len(df_weather)} righe dal {df_weather.index.min()} al {df_weather.index.max()}")
        return df_weather
    except Exception as e:
        print(f"Errore durante il caricamento o la pulizia dei dati climatici: {e}")
        return pd.DataFrame()


# --- MAIN SCRIPT ---
if __name__ == "__main__":
    print("--- Avvio Script di Previsione Consumi Energetici ---")

    # --- 2. ACQUISIZIONE DATI ---

    # 2.1. Dati API in Tempo Reale
    print("\n--- Recupero Stato Corrente dei Sensori (Home Assistant API) ---")
    current_sensor_states = {}
    for name, entity_id in ENTITY_IDS.items():
        state_data = get_current_state(entity_id)
        if state_data:
            current_sensor_states[name] = state_data['state']
            last_changed_utc = pd.to_datetime(state_data['last_changed']).astimezone(pytz.utc)
            last_updated_utc = pd.to_datetime(state_data['last_updated']).astimezone(pytz.utc)
            last_changed_local = last_changed_utc.astimezone(LOCAL_TIMEZONE)
            last_updated_local = last_updated_utc.astimezone(LOCAL_TIMEZONE)
            print(
                f"  {name.capitalize()}: {state_data['state']} (Ultimo cambio: {last_changed_local.strftime('%Y-%m-%d %H:%M:%S %Z%z')}, Ultimo aggiornamento: {last_updated_local.strftime('%Y-%m-%d %H:%M:%S %Z%z')})")
    print(f"\nDati in tempo reale raccolti: {current_sensor_states}")

    # 2.2. Dati API Storici (es. ultimi 3 mesi)
    print("\n--- Recupero Dati Storici dai Sensori (Home Assistant API) ---")
    end_date_api_historical = datetime.now(UTC)
    start_date_api_historical = end_date_api_historical - timedelta(days=120)

    api_sensor_dataframes = []
    for name, entity_id in ENTITY_IDS.items():
        raw_historical_data = get_historical_data(entity_id, start_date_api_historical, end_date_api_historical)
        df_sensor = clean_and_process_data_to_df(raw_historical_data, name)
        if not df_sensor.empty:
            api_sensor_dataframes.append(df_sensor)
        else:
            print(f"Non è stato possibile ottenere dati storici validi per {name} da API. Saltando.")

    df_api_historical = combine_all_sensor_data(api_sensor_dataframes, target_freq='h')
    # Rimuovi timezone (converti a naive datetime) dall'indice datetime
    df_to_save = df_api_historical.copy()
    df_to_save.index = df_to_save.index.tz_localize(None)

    # Ora puoi salvare in Excel senza errori
    df_to_save.to_excel("df_api_historical.xlsx", index=True)
    print("Salvato df_api_historical.xlsx senza timezone nell'indice.")


