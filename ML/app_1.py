import streamlit as st
import numpy as np
import joblib
import pandas as pd 

df = pd.read_csv("../data/spotify_clean.csv")

# Cargar el modelo entrenado y el PCA
rf_model = joblib.load("modelo_random_forest_1.pkl")
normalizer = joblib.load("normalizer1.pkl")

# Incializaer valores como variables de estado

default_values = {
    "danceability": 0.5, "energy": 0.5, "speechiness": 0.5, "acousticness": 0.5,
    "instrumentalness": 0.0, "liveness": 0.5, "valence": 0.5, "tempo": 120.0,
    "duration_s": 200.0, "key": "C", "time_signature": "4/4", "mode": "Major",
    "resultado": None
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value
        
        
# Función para predecir la popularidad de la canción
def predecir_popularidad(features):
    
    #st.write(features)
    features = np.array(features).reshape(1, -1)  # Convertir a la forma correcta
    
    # Normalizar las características de entrada
    features_norm = normalizer.transform(features)
    
    # Predecir con Random Forest
    prediccion = rf_model.predict(features_norm)
    return prediccion[0]

# Función para transformar las entradas del usuario
def transformar_entradas(key, time_signature, mode):
    # Convertir 'key' a número
    
    # Get unique values from the 'key' column
    list_of_keys = df['key'].unique()

    # Create a dictionary to map each key to a number
    key_mapping = {key: i for i, key in enumerate(list_of_keys)}
    key_numeric = key_mapping.get(key, -1) # Si el valor no está en la lista, asigna un valor predeterminado

    # Convertir 'time_signature' a número
    list_of_time_signatures = df['time_signature'].unique() # Asegúrate de que esta lista sea la misma que usaste en el entrenamiento
    time_signature_mapping = {time_signature: i for i, time_signature in enumerate(list_of_time_signatures)}
    time_signature_numeric = time_signature_mapping.get(time_signature, -1)

    # Convertir 'mode' a número
    mode_numeric = 1 if mode == 'Major' else 0

    return key_numeric, time_signature_numeric, mode_numeric

# Interfaz de la app
st.title("🎵 Predicción de Popularidad de Canciones con Random Forest")

st.subheader("Ingrese las características de la canción:")

# Widgets con valores guardados en session_state
st.session_state.danceability = st.slider("Danceability", 0.0, 1.0, st.session_state.danceability)
st.session_state.energy = st.slider("Energy", 0.0, 1.0, st.session_state.energy)
st.session_state.speechiness = st.slider("Speechiness", 0.0, 1.0, st.session_state.speechiness)
st.session_state.acousticness = st.slider("Acousticness", 0.0, 1.0, st.session_state.acousticness)
st.session_state.instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, st.session_state.instrumentalness)
st.session_state.liveness = st.slider("Liveness", 0.0, 1.0, st.session_state.liveness)
st.session_state.valence = st.slider("Valence", 0.0, 1.0, st.session_state.valence)
st.session_state.tempo = st.slider("Tempo (BPM)", 50.0, 200.0, st.session_state.tempo)
st.session_state.duration_s = st.slider("Duration (seconds)", 14.0, 6000.0, st.session_state.duration_s)

st.session_state.key = st.selectbox("Key", ['C#', 'F#', 'C', 'F', 'G', 'E', 'D#', 'G#', 'D', 'A#', 'A', 'B'], index=['C#', 'F#', 'C', 'F', 'G', 'E', 'D#', 'G#', 'D', 'A#', 'A', 'B'].index(st.session_state.key))
st.session_state.time_signature = st.selectbox("Time Signature", ['4/4', '5/4', '3/4', '1/4', '0/4'], index=['4/4', '5/4', '3/4', '1/4', '0/4'].index(st.session_state.time_signature))
st.session_state.mode = st.selectbox("Mode", ['Major', 'Minor'], index=['Major', 'Minor'].index(st.session_state.mode))

# Botón para predecir
if st.button("Predecir Popularidad"):
    key_numeric, time_signature_numeric, mode_numeric = transformar_entradas(st.session_state.key, st.session_state.time_signature, st.session_state.mode)
    
    features = [
        st.session_state.acousticness, st.session_state.danceability, st.session_state.energy,
        st.session_state.instrumentalness, key_numeric, st.session_state.liveness,
        mode_numeric, st.session_state.speechiness, st.session_state.tempo,
        time_signature_numeric, st.session_state.valence, st.session_state.duration_s
    ]

    # Guardar el resultado en session_state
    st.session_state.resultado = predecir_popularidad(features)

# Mostrar el resultado solo si ya se ha ejecutado la predicción
if st.session_state.resultado is not None:
    st.write("🎶 **Resultado:**", st.session_state.resultado)
