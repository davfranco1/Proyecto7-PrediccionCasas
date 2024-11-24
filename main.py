import streamlit as st
import pandas as pd
import pickle
import time

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
st.title("🏠 Predicción de Precios de Casas en Madrid con Machine Learning")
st.write("Esta aplicación permite predecir el precio del alquiler de una casa basándonos en sus características.")

# Mostrar una imagen llamativa
st.image(
    "images/header.png",  # URL de la imagen
    #caption="Tu próxima casa está aquí.",
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
    propertytype = st.selectbox("Tipo de Casa", ["flat", "studio", "duplex", "penthouse", "chalet", "countryHouse"], help="Selecciona el tipo de casa.")
    exterior = st.selectbox("Exterior", ["True", "False", "Sin información"], help="Elige característica de la casa.")
    hasLift = st.selectbox("Tiene Ascensor", ["True", "False"], help="Elige característica de la casa.")
    municipality = st.selectbox("Municipio", ['Madrid', 'San Sebastián de los Reyes', 'Villamanrique de Tajo',
       'Rascafría', 'Manzanares el Real', 'Miraflores de la Sierra',
       'Galapagar', 'Arganda', 'San Lorenzo de el Escorial',
       'Villanueva del Pardillo', 'Aranjuez', 'Las Rozas de Madrid',
       'Navalcarnero', 'Alcalá de Henares', 'El Escorial', 'Leganés',
       'Coslada', 'Torrejón de Ardoz', 'Camarma de Esteruelas',
       'Alcorcón', 'Valdemoro', 'Collado Villalba', 'Getafe',
       'Paracuellos de Jarama', 'El Molar', 'Parla', 'Tres Cantos',
       'Quijorna', 'Valdemorillo', 'Pedrezuela', 'Daganzo de Arriba',
       'Guadarrama', 'Cobeña', 'El Álamo', 'Algete', 'Rivas-Vaciamadrid',
       'Pinto', 'Los Santos de la Humosa', 'San Fernando de Henares',
       'Aldea del Fresno', 'Fuenlabrada', 'Mataelpino', 'Villa del Prado',
       'Los Molinos', 'Colmenar Viejo', 'Móstoles', 'Navalafuente',
       'Robledo de Chavela', 'Villaviciosa de Odón', 'Pozuelo de Alarcón',
       'Bustarviejo', 'Collado Mediano', 'Chinchón', 'Colmenarejo',
       'Loeches', 'Sevilla la Nueva', 'Campo Real', 'Torrelaguna',
       'Villalbilla', 'Alcobendas'])

with col2:
    rooms = st.number_input("Número de Habitaciones", min_value=1, max_value=6, step=1)
    bathrooms = st.number_input("Número de Baños", min_value=1, max_value=3, step=1)
    size = st.number_input("Área en m²", min_value=50, max_value=100, step=10)
    floor = st.number_input("Planta", min_value=0, max_value=14, step=1)
    distance = st.number_input("Distancia al centro", min_value=0, max_value=10000, step=1)

# Botón para realizar la predicción
if st.button("💡 Predecir Precio"):
    # Crear DataFrame con los datos ingresados
    nueva_casa = pd.DataFrame({
    'propertyType': [propertytype], 
    'size': [size],
    'exterior': [exterior],
    'rooms': [rooms],
    'bathrooms': [bathrooms],
    'municipality':[municipality],
    'distance':[distance],
    'floor':[floor],
    'hasLift':[hasLift]
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
    prediction = round(model.predict(df_pred)[0],0)

    # Mostrar el resultado
    with st.spinner('Estamos calculando el valor del alquiler...'):
        time.sleep(3)
    st.success(f"💵 El precio estimado del alquiler de la casa es de {prediction}€")

# Pie de página
st.markdown(
    """
    ---
    **Proyecto creado con el potencial de la ciencia de datos.**  
    Desarrollado con ❤️ usando Streamlit.
    """,
    unsafe_allow_html=True,
)