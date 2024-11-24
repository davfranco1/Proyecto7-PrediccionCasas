import streamlit as st
import pandas as pd
import pickle

from category_encoders import TargetEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Configurar la página de Streamlit
st.set_page_config(
    page_title="Predicción de Precios de Casas",
    page_icon="🏠",
    layout="centered",
)

# Título y descripción
st.title("🏠 Predicción de Precios de Casas con Machine Learning")
st.write("Usa esta aplicación para predecir el precio de una casa basándote en sus características. ¡Sorpréndete con la magia de los datos! 🚀")

# Mostrar una imagen llamativa
st.image(
    "images/header.png",  # URL de la imagen
    caption="Tu próxima casa está aquí.",
    use_container_width=True,
)

# Cargar los modelos y transformadores entrenados
def load_models():
    with open('modelos/onehot_encoder.pkl', 'rb') as f:
        onehot_encoder = pickle.load(f)

    with open('modelos/target_encoder.pkl', 'rb') as f:
        target_encoder = pickle.load(f)

    with open('modelos/standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('modelos/rf_regressor.pkl', 'rb') as f:
        model = pickle.load(f)

    return onehot_encoder, target_encoder, scaler, model

onehot_encoder, target_encoder, scaler, model = load_models()

# Formularios de entrada
st.header("🔧 Características de la casa")
col1, col2 = st.columns(2)

with col1:
    neighborhood = st.selectbox("Barrio", ["A", "B", "C", "D"], help="Selecciona el barrio de la casa.")
    house_type = st.selectbox("Tipo de Casa", ["Detached", "Semi-Detached"], help="Elige el tipo de casa.")

with col2:
    rooms = st.number_input("Número de Habitaciones", min_value=1, max_value=10, value=3, step=1)
    area = st.number_input("Área en m²", min_value=50, max_value=500, value=120, step=10)

# Botón para realizar la predicción
if st.button("💡 Predecir Precio"):
    # Crear DataFrame con los datos ingresados
    nueva_casa = pd.DataFrame({
    'propertyType': ["Flat"], 
    'size': [60],
    'exterior': ["True"],
    'rooms': [2],
    'bathrooms': [1],
    'municipality':["Madrid"],
    'distance':[2000],
    'floor':[2],
    'hasLift':["True"]
    })

    df_new = pd.DataFrame(nueva_casa)
    df_new

    df_pred = df_new.copy()
    col_num=["size", "distance"]
    df_pred[col_num] = scaler.transform(df_pred[col_num])

    df_pred['hasLift'] = df_pred['hasLift'].map({
        "True": 1, 
        "False": 0, 
        'Sin información': -1
    })

    df_pred['exterior'] = df_pred['exterior'].map({
        "True": 1, 
        "False": 0, 
        'Sin información': -1
    })

    df_pred['floor'] = df_pred['floor'].map({
        'st': -2, 
        'ss': -1,
        'bj': 0,
        'en': 0.5, 
        'Sin información': -3
    }).fillna(df_pred['floor']) 

    # Transformar columna con OneHot
    df_pred["municipality"] = target_encoder.transform(df_pred["municipality"])

    # Transformar columna con Target
    encoded_array = onehot_encoder.transform(df_pred[["propertyType"]])

    # Crear un DF con las nuevas columnas
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=onehot_encoder.get_feature_names_out(["propertyType"]),
        index=df_pred.index
    )

    # Concatenar con el DF original
    df_pred = pd.concat([df_pred.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Drop de la columna original que ya no necesitamos
    df_pred.drop("propertyType", axis=1, inplace=True)

    df_pred.rename(columns= {"size":"size_standard", "distance":"distance_standard"}, inplace=True)

    df_pred = df_pred[['exterior', 'rooms', 'bathrooms', 'floor', 'hasLift',
        'propertyType_chalet', 'propertyType_countryHouse',
        'propertyType_duplex', 'propertyType_flat', 'propertyType_penthouse',
        'propertyType_studio', 'municipality', 'size_standard',
        'distance_standard']]

    # Realizar la predicción
    prediction = model.predict(df_pred)[0]

    # Mostrar el resultado
    st.success(f"💵 El precio estimado de la casa es: ${prediction}")
    st.balloons()

# Pie de página
st.markdown(
    """
    ---
    **Proyecto creado con el potencial de la ciencia de datos.**  
    Desarrollado con ❤️ usando Streamlit.
    """,
    unsafe_allow_html=True,
)