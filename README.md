# Proyecto 7: Predicción de precios de alquileres residenciales en la Comunidad de Madrid

![imagen](images/header.png)


## Planteamiento del problema: **predicción de los precios la vivienda de alquiler.**

- La predicción de precios de bienes inmuebles es un área clave en la intersección de los negocios y la ciencia de datos. En este proyecto, abordarás el desafío de estimar el precio de las casas. Trabajaras con un conjunto de datos real, que contiene información detallada sobre propiedades en Madrid, como su tamaño, ubicación, número de habitaciones, tipo de propiedad, y más. 


## Objetivos del Proyecto

- El mercado inmobiliario es dinámico y está influenciado por múltiples variables, como la ubicación, las características de la propiedad y las condiciones económicas. Un modelo predictivo preciso puede ser una herramienta poderosa para agentes inmobiliarios, compradores y vendedores.

- Este proyecto se sumerge en la complejidad de estos factores y pretende transformarlos en conocimiento útil para la toma de decisiones. Su objetivo principal es predecir el precio del alquiler de una casa.

1. **Preprocesamiento**: Abarca todas las etapas de preparación de los datos: EDA, gestión de nulos, encoding, outliers y estandarización.

2. **Modelos predictivos**: Selección y prueba de los modelos más precisos usando Scikitlearn.

3. **Presentación de los datos**: Utilizar Streamlit como plataforma para la consulta sencilla de las predicciones.


## Estructura del repositorio

El proyecto está construido de la siguiente manera:

- **datos/**: Carpeta que contiene archivos `.csv`, `.json` o `.pkl` generados durante la captura y tratamiento de los datos.

- **images/**: Carpeta que contiene archivos de imagen generados durante la ejecución del código o de fuentes externas.

- **notebooks/**: Carpeta que contiene los archivos `.ipynb` utilizados en la captura y tratamiento de los datos. Están numerados para su ejecución secuencial, y contenidos dentro de 3 carpetas, una para cada modelo, conteniendo cada una de ellas:
  - `1_EDA`
  - `2_Encoding`
  - `3_Outliers`
  - `4_Estandarización`
  - `5_Modelos`
  - `6_Predicciones`

- **src/**: Carpeta que contiene los archivos `.py`, con las funciones y variables utilizadas en los distintos notebooks.

- `.gitignore`: Archivo que contiene los archivos y extensiones que no se subirán a nuestro repositorio, como los archivos .env, que contienen contraseñas.


## Lenguaje, librerías y temporalidad
- El proyecto fué elaborado con Python 3.9 y múltiples librerías de soporte:

*Librerías para el tratamiento de datos*
- [Pandas](https://pandas.pydata.org/docs/)
- [Numpy](https://numpy.org/doc/)

*Librerías para captura de datos*
- [Selenium](https://selenium-python.readthedocs.io)
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Requests](https://pypi.org/project/requests/)

*Librerías para gestión de tiempos*
- [Time](https://docs.python.org/3/library/time.html)
- [tqdm](https://numpy.org/doc/)

*Librerías para graficar y mapas*
- [Plotly](https://plotly.com/python/)
- [Seaborn](https://seaborn.pydata.org)
- [Matplotlib](https://matplotlib.org/stable/index.html)
- [Folium](https://python-visualization.github.io/folium/latest/)

*Librería para gestionar tokens y contraseñas*
- [DotEnv](https://pypi.org/project/python-dotenv/)

*Librería para controlar parámetros del sistema*
- [Sys](https://docs.python.org/3/library/sys.html)

*Librería para controlar ficheros*
- [os](https://docs.python.org/3/library/os.html)

*Librería para generar aplicaciones basadas en Python*
- [streamlit](https://docs.streamlit.io)

*Librería para creación de modelos de Machine Learning*
- [scikitlearn](https://scikit-learn.org/stable/)

*Librería para creación de iteradores (utilizada para combinaciones)*
- [itertools](https://docs.python.org/3/library/itertools.html)

*Librería para la gestión de avisos*
- [warnings](https://docs.python.org/3/library/warnings.html)


- Este proyecto es funcional a fecha 24 de noviembre de 2024.


## Instalación

1. Clona el repositorio
   ```sh
   git clone https://github.com/davfranco1/Proyecto5-ProyectoLibre-PreciosAlquileresMadrid.git
   ```

2. Instala las librerías que aparecen en el apartado anterior. Utiliza en tu notebook de Jupyter:
   ```sh
   pip install nombre_librería
   ```

3. Cambia la URL del repositorio remoto para evitar cambios al original.
   ```sh
   git remote set-url origin usuario_github/nombre_repositorio
   git remote -v # Confirma los cambios
   ```

4. Ejecuta el código en los notebooks, modificándolo si es necesario.

5. Para utilizar la app de Streamlit, copia el repositorio, abre una terminal en la carpeta base, y ejecuta el comando `streamlit run main.py`, que abrirá un navegador donde se ejecuta automáticamente el código. Recuerda que antes debes haber instalado la librería `pip install streamlit`.


## Conclusiones

- Hemos creado tres modelos para la predicción de los precios del alquiler. La principal dificultar con la que nos hemos encontrado es el overfitting en los modelos. El overfitting ocurre cuando un modelo se "aprende de memoria" los datos de entrenamiento, entonces, cuando entra un nuevo dato desconocido, es incapaz de predecirlo con precisión.

- Para disminuir el overfitting, tenemos varias alternativas, incluyendo: disminuir el número de variables predictoras, aumentar la cantidad de datos o disminuir la profundidad de los árboles, cuando corresponda, de modo que sean menos específicos.

- Tras numerosas iteraciones, nos quedaremos con el Modelo 2, que tiene unas métricas para el R2 de 0.53 y 0.59 respectivamente para el train y el test, y un RMSE de 39.9 y 39.2.




## Próximos pasos

- Corresponde seguir trabajando sobre el modelo para mejorar estas métricas, y conseguir predecir mejor el precio del alquiler de las viviendas. El primer paso sería intentar obtener más datos, de cara a mejorar el modelo.

- También puede mejorarse la app de Streamlit, añadiendo las explicaciones del modelo en esta página.


## Autor

David Franco - [LinkedIn](https://linkedin.com/in/franco-david)

Enlace del proyecto: [https://github.com/davfranco1/Proyecto7-PrediccionCasas](https://github.com/davfranco1/Proyecto7-PrediccionCasas)
