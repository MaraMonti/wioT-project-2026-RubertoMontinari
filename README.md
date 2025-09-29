#wot-project-2026-rubertomontinari-gateway-ml

**⚡ **Previsione della domanda energetica della casa intelligente****


**A) Descrizione Generale del Progetto**

Il progetto Smart Home Energy Demand Prediction mira a sviluppare un sistema avanzato per la previsione accurata del consumo energetico giornaliero di un'abitazione.

La problematica affrontata è la necessità di ottimizzare i consumi energetici e mitigare gli sprechi, un fattore cruciale sia a livello economico che ambientale.

La soluzione proposta si basa sull'applicazione di tecniche di Machine Learning (ML) avanzate, in grado di identificare pattern complessi nel consumo. Il modello sfrutta:

Un dataset storico fornito.

Dati in tempo reale acquisiti tramite API (consumo, temperatura, umidità).

L'integrazione di dataset esterni (es. dati meteorologici da Tomorrow.io) per migliorare l'accuratezza e la generalizzazione.

**B) Architettura del Sistema**

Il sistema è strutturato come un'applicazione web interattiva basata su Streamlit, che funge da interfaccia per un flusso di dati e calcoli predittivi.

Architettura:

Fonti dei dati: Raccolta dati da diverse sorgenti:

Dataset iniziale: Dataset storico fornito per l'abitazione di Lecce.

API Home Assistant: API Rest per l'acquisizione di dati istantanei di consumo, temperatura e umidità in tempo reale.

Tomorrow.io API: API per dati meteorologici (previsioni e storici).

Modulo Elaborazione Dati: Modulo di pre-elaborazione (data_preprocessing.py) responsabile della pulizia, unificazione (fusione di ~380.000 record da fonti multiple come Kaggle e UMass) e Feature Engineering (creazione di variabili temporali come hour, day_of_week, is_weekend).

Streamlit Application (Frontend): Interfaccia utente che carica il modello ML, visualizza i dati storici, mostra i dati in tempo reale e genera le previsioni.


<img width="877" height="514" alt="Image" src="https://github.com/user-attachments/assets/a774f1ff-40f4-4569-bb08-25405fbea028" />

---

**C) Repository e Componenti**
Il progetto è suddiviso in diversi repository, ciascuno per un componente specifico. Di seguito l'elenco dei repository con i rispettivi link:
| Componente | Ruolo | Repository Link |
| :--- | :--- | :--- |
| **Edge Device** | Rilevazione locale (simulata o reale) dei dati. | **[https://github.com/MaraMonti/wot-project-2025-rubertomontinari-edge-device]** |
| **Gateway / ML Service** | Contiene il Modello ML e l'App Streamlit (questo repository). | **[https://github.com/MaraMonti/wot-project-2025-RubertoMontinari-gateway-ml]** |
| **Front-end / App** | (Se separato) Interfaccia utente finale. | **[LINK REPO FRONT-END PLACEHOLDER]** |



---

 **D) Dettaglio Componente: Gateway / ML Service (Streamlit App)**

Questo repository contiene il cuore del sistema di previsione, incluso il modello di Machine Learning addestrato e l'applicazione web interattiva sviluppata in Python/Streamlit.

Modello di Previsione
Algoritmo: XGBoost. È stato scelto per la sua robustezza e capacità di catturare relazioni non lineari nei dati di consumo.

Addestramento: Il modello è addestrato su un dataset arricchito e pulito di circa 380.000 record.

Obiettivo: Prevedere il consumo totale di energia dell'abitazione per l'intera giornata (a partire da dati storici orari).

Funzionalità dell'applicazione Streamlit
L'interfaccia utente fornisce diverse pagine gestite da moduli specifici (predict_from_api.py, dati_home.py, forecast_dashboard.py):

Predizione Manuale: Permette di ottenere previsione istantanea basata su input manuali (es. temperatura e umidità).

Predizione da Meteo API: Acquisisce previsioni future (24-48 ore) di temperatura e umidità da Tomorrow.io per generare previsioni proattive del consumo.

Dati Abitazione: Monitoraggio in tempo reale del consumo energetico, temperatura e umidità tramite l'API di Home Assistant.

Dashboard Previsioni: Visualizzazione di metriche di valutazione del modello (MAE, RMSE, R²) e confronti tra valori reali e previsti.

Tecnologie Utilizzate

<img width="888" height="385" alt="Image" src="https://github.com/user-attachments/assets/954e7f44-052c-4ed4-bc7e-f3b4f946caa7" />


### ⚙️ Istruzioni per l'Avvio dell'Interfaccia Utente

Per avviare l'applicazione Streamlit in locale (il frontend del progetto):

1.  **Clona il repository:**
    ```bash
    git clone [https://github.com/MaraMonti/wot-project-2025-rubertomontinari-gateway-ml]
    ```
2.  **Installa le dipendenze:** 
    ```bash
    pip install -r requirements.txt
    ```
3.  **Avvia l'App:**
    ```bash
    streamlit run main.py
    ```
L'applicazione sarà accessibile nel browser all'indirizzo `http://localhost:8501`.
