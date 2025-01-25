import streamlit as st
import numpy as np
import joblib

# Cargar el modelo entrenado y el PCA
rf_model = joblib.load("modelo_random_forest.pkl")
pca = joblib.load("pca_model.pkl")  # Asegúrate de que el archivo pca_model.pkl sea el que usaste durante el entrenamiento
normalizer = joblib.load("normalizer.pkl")

# Función para predecir la popularidad de la canción
def predecir_popularidad(features):
    
    features = np.array(features).reshape(1, -1)  # Convertir a la forma correcta
    
    # Normalizar las características de entrada
    features_norm = normalizer.transform(features)
    
    # Aplicar el PCA para reducir la dimensionalidad
    features_pca = pca.transform(features)
    
    # Predecir con Random Forest
    prediccion = rf_model.predict(features_pca)
    return prediccion

# Función para transformar las entradas del usuario
def transformar_entradas(key, time_signature, mode):
    # Convertir 'key' a número
    list_of_keys = ['C#', 'F#', 'C', 'F', 'G', 'E', 'D#', 'G#', 'D', 'A#', 'A', 'B']  # Asegúrate de que esta lista sea la misma que usaste en el entrenamiento
    key_mapping = {key: i for i, key in enumerate(list_of_keys)}
    key_numeric = key_mapping.get(key, -1)  # Si el valor no está en la lista, asigna un valor predeterminado

    # Convertir 'time_signature' a número
    list_of_time_signatures = ['4/4', '5/4', '3/4', '1/4', '0/4']  # Asegúrate de que esta lista sea la misma que usaste en el entrenamiento
    time_signature_mapping = {time_signature: i for i, time_signature in enumerate(list_of_time_signatures)}
    time_signature_numeric = time_signature_mapping.get(time_signature, -1)

    # Convertir 'mode' a número
    mode_numeric = 1 if mode == 'Major' else 0

    return key_numeric, time_signature_numeric, mode_numeric

# Interfaz de la app
st.title("🎵 Predicción de Popularidad de Canciones con Random Forest")

st.subheader("Ingrese las características de la canción:")

# Inputs (ajusta según las características de tu dataset)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
loudness = st.slider("Loudness (dB)", -60.0, 0.0, -30.0)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.5)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
liveness = st.slider("Liveness", 0.0, 1.0, 0.5)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo (BPM)", 50.0, 200.0, 120.0)
duration_s = st.slider("Duration (seconds)", 14.0, 6000.0, 200.0)

# Inputs adicionales (key, time_signature, mode)
key = st.selectbox("Key", ['C#', 'F#', 'C', 'F', 'G', 'E', 'D#', 'G#', 'D', 'A#', 'A', 'B'])
time_signature = st.selectbox("Time Signature", ['4/4', '5/4', '3/4', '1/4', '0/4'])
mode = st.selectbox("Mode", ['Major', 'Minor'])

# Botón para predecir
if st.button("Predecir Popularidad"):
    # Transformar las características
    key_numeric, time_signature_numeric, mode_numeric = transformar_entradas(key, time_signature, mode)
    
    # Crear las características de entrada
    features = [danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo,
                key_numeric, time_signature_numeric, mode_numeric, duration_s]
    
    # Predecir la popularidad
    resultado = predecir_popularidad(features)
    
    # Mostrar el resultado
    st.write("🎶 **Resultado:**", resultado)
