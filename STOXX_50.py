import streamlit as st
import pandas as pd
import pydeck as pdk
import os

# Configuración de la página
st.set_page_config(page_title="Mapa 3D de Empresas por País Europeo", layout="wide")
logo_path = "images/STEP 1-4.png"  # Reemplaza con la ruta real a tu imagen
st.image(logo_path, width=350)
st.title("EuroStoxx 50: Inversión con visión")
st.subheader("Construyendo un mañana rentable y sostenible.")
st.markdown(
    """
    <div style="margin-left: 0px; margin-right: 40px; font-size: 18px; text-align: justify;">
        El EuroStoxx 50 es el índice bursátil de referencia en Europa, compuesto por las 50 empresas líderes de la eurozona. 
        Representa sectores clave como tecnología, energía, finanzas e industria, sirviendo como un termómetro del desempeño 
        económico en la región. Su relevancia va más allá de la rentabilidad: en un mundo donde la inversión sostenible gana 
        protagonismo, muchas de las empresas que lo conforman están adoptando estrategias ESG para garantizar un crecimiento 
        responsable. Para los inversores, el EuroStoxx 50 no solo ofrece diversificación y estabilidad, sino también la 
        oportunidad de alinear sus decisiones financieras con un futuro más sostenible.
    </div>
    """,
    unsafe_allow_html=True,
)
# Diccionario de coordenadas de países europeos
EUROPEAN_COUNTRY_COORDS = {
    "España": (40.4168, -3.7038),
    "Reino Unido": (55.3781, -3.4360),
    "Francia": (46.2276, 2.2137),
    "Alemania": (51.1657, 10.4515),
    "Italia": (41.8719, 12.5674),
    "Portugal": (39.3999, -8.2245),
    "Países Bajos": (52.1326, 5.2913),
    "Bélgica": (50.5039, 4.4699),
    "Suiza": (46.8182, 8.2275),
    "Austria": (47.5162, 14.5501),
    "Suecia": (60.1282, 18.6435),
    "Noruega": (60.4720, 8.4689),
    "Dinamarca": (56.2639, 9.5018),
    "Finlandia": (61.9241, 25.7482),
    "Grecia": (39.0742, 21.8243),  
    "Irlanda": (53.1424, -7.6921),
    "Polonia": (51.9194, 19.1451),
    "República Checa": (49.8175, 15.4730),
    "Hungría": (47.1625, 19.5033),
    "Rumania": (45.9432, 24.9668),
    "Bulgaria": (42.7339, 25.4858),
    "Eslovaquia": (48.6690, 19.6990),
    "Eslovenia": (46.1512, 14.9955),
    "Croacia": (45.1000, 15.2000),
    "Serbia": (44.0165, 21.0059),
    "Bosnia y Herzegovina": (43.9159, 17.6791),
    "Montenegro": (42.7087, 19.3744),
    "Macedonia del Norte": (41.6086, 21.7453),
    "Albania": (41.1533, 20.1683),
    "Ucrania": (48.3794, 31.1656),
    "Bielorrusia": (53.7098, 27.9534),
    "Moldavia": (47.4116, 28.3699),
    "Lituania": (55.1694, 23.8813),
    "Letonia": (56.8796, 24.6032),
    "Estonia": (58.5953, 25.0136),
    "Luxemburgo": (49.8153, 6.1296),
    "Malta": (35.9375, 14.3754),
    "Chipre": (35.1264, 33.4299),
    "Islandia": (64.9631, -19.0208),
}

# Función para cargar datos desde archivo XLS
@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")
            return pd.DataFrame()
    else:
        st.error(f"El archivo {file_path} no existe")
        return pd.DataFrame()

# Ruta al archivo XLS (cambia esta ruta por la ubicación de tu archivo)
xls_path = "empresas_europeas.xlsx"  # Reemplaza con la ruta real a tu archivo

# Cargar los datos
df = load_data(xls_path)

# Mostrar datos cargados
# st.subheader("Datos de empresas cargados")
# with st.expander("Ver datos originales"):
 #   st.dataframe(df)

# Verificar que las columnas necesarias existen
required_columns = ["empresa", "sector", "pais"]
if not all(col in df.columns for col in required_columns):
    st.error("El archivo debe contener las columnas: 'empresa', 'sector' y 'pais'")
    st.stop()

# Verificar y procesar datos
if not df.empty:
    # Crear diccionario de colores por sector
    sectors = df["sector"].unique()
    color_dict = {}
    # Generar colores para cada sector
    for i, sector in enumerate(sectors):
        r = (i * 50) % 255
        g = (i * 100) % 255
        b = (i * 150) % 255
        color_dict[sector] = [r, g, b, 200]
    
    # Agregar coordenadas de países
    # st.write("Procesando coordenadas geográficas...")
    progress_bar = st.progress(0)
    
    # Añadir coordenadas al DataFrame
    geo_df = df.copy()
    geo_df["lat"] = geo_df["pais"].map(lambda x: EUROPEAN_COUNTRY_COORDS.get(x, (None, None))[0])
    geo_df["lon"] = geo_df["pais"].map(lambda x: EUROPEAN_COUNTRY_COORDS.get(x, (None, None))[1])
    
    # Mostrar advertencia si hay países sin coordenadas
    missing_countries = df[~df["pais"].isin(EUROPEAN_COUNTRY_COORDS.keys())]["pais"].unique()
    if len(missing_countries) > 0:
        st.warning(f"No se encontraron coordenadas para los siguientes países: {', '.join(missing_countries)}")
    
    # Filtrar filas sin coordenadas
    geo_df = geo_df.dropna(subset=["lat", "lon"])
    
    if geo_df.empty:
        st.error("No se pudieron obtener coordenadas para los países indicados.")
        st.stop()
    
    # Actualizar barra de progreso
    progress_bar.progress(1.0)
    
    # Agregar columna de color basada en sector
    geo_df["color"] = geo_df["sector"].map(color_dict)
    
    # Modificar datos para visualización
    # Contar empresas por ubicación y sector para determinar altura
    location_counts = geo_df.groupby(["pais", "sector", "lat", "lon"]).size().reset_index(name="count")
    
    # Ajustar datos para la visualización
    chart_data = []
    height_multiplier = 10000  # Multiplicador para la altura visual
    
    for _, row in location_counts.iterrows():
        country = row["pais"]
        sector = row["sector"]
        companies = geo_df[(geo_df["pais"] == country) & (geo_df["sector"] == sector)]["empresa"].tolist()
        
        chart_data.append({
            "lat": row["lat"],
            "lng": row["lon"],
            "height": row["count"] * height_multiplier,
            "color": color_dict[sector],
            "country": country,
            "sector": sector,
            "companies": companies,
            "count": row["count"]
        })
    
    # Crear visualización con PyDeck
    # st.write("### Mapa 3D de Empresas por País y Sector")
    
    # Controles de visualización
    col1, col2 = st.columns(2)
    with col1:
        height_factor = st.slider("Altura de las barras", min_value=10, max_value=50, value=10, step=1)
    with col2:
        radius = st.slider("Radio de visualización", min_value=15000, max_value=25000, value=15000, step=100)
    
    # Ajustar altura basada en el control deslizante
    for item in chart_data:
        item["height"] = item["count"] * height_multiplier * height_factor
    
    # Definir la capa de columnas 3D
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=chart_data,
        get_position=["lng", "lat"],
        get_elevation="height",
        elevation_scale=1,
        radius=radius,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )
    
    # Calcular centro del mapa para Europa
    center_lat = 48.6  # Aproximadamente el centro de Europa
    center_lon = 10.0
    
    # Configurar vista inicial
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=3.5,
        pitch=45,
    )
    
    # Configurar el tooltip para mostrar información al pasar el cursor
    tooltip = {
        "html": "<b>País:</b> {country}<br><b>Sector:</b> {sector}<br><b>Número de empresas:</b> {count}<br><b>Empresas:</b> {companies}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    
    # Renderizar el mapa
    r = pdk.Deck(
        layers=[column_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9",
    )
    
    st.pydeck_chart(r)
    
    # Mostrar leyenda
    st.write("### Leyenda de Sectores")
    legend_cols = st.columns(min(4, len(color_dict)))  # Máximo 4 columnas
    
    for i, (sector, color) in enumerate(color_dict.items()):
        col_idx = i % len(legend_cols)
        with legend_cols[col_idx]:
            st.markdown(
                f"<div style='background-color: rgba({color[0]}, {color[1]}, {color[2]}, {color[3]/255}); "
                f"height: 20px; width: 20px; display: inline-block; margin-right: 5px;'></div> {sector}",
                unsafe_allow_html=True
            )
    
    # Mostrar tablas de conteo por país y sector en columnas
st.markdown("<hr style='border:1px solid lightgray'>", unsafe_allow_html=True)
st.write("### Empresas por País y Sector")
# Dividir en tres columnas: tabla de países, texto, tabla de sectores
col1, col2, col3 = st.columns([0.42, 0.3, 0.5])  # Ajusta los anchos según sea necesario

with col2:
    st.write("#### Empresas por País")
    country_counts = geo_df.groupby("pais").size().reset_index(name="count")
    country_counts = country_counts.sort_values("count", ascending=False)
    country_counts.index = range(1, len(country_counts) + 1)  # Ajustar índice para que comience en 1
    st.dataframe(country_counts)

with col3:
    st.markdown(
        """
        <div style="margin-left: 20px; margin-right: 20px;">
            <h4 style="font-size: 24px;">Análisis:</h4>
            <p style="font-size: 16px; text-align: justify;">
            Esta distribución refleja la estructura económica tradicional europea, donde el poder financiero y la manufactura industrial, particularmente automotriz, han sido históricamente predominantes. De cara al futuro, esta concentración sugiere que el eje franco-alemán seguirá siendo el motor económico de Europa, aunque la relativamente baja representación de sectores tecnológicos y emergentes (solo 2 empresas de semiconductores) podría indicar un desafío estructural para la competitividad europea en la economía digital global. Para mantener su relevancia económica, Europa necesitará fortalecer su posición en sectores de alto crecimiento y tecnologías avanzadas, equilibrando su fortaleza tradicional en finanzas e industria manufacturera.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col1:
    st.write("#### Empresas por Sector")
    sector_counts = geo_df.groupby("sector").size().reset_index(name="count")
    sector_counts = sector_counts.sort_values("count", ascending=False)
    sector_counts.index = range(1, len(sector_counts) + 1)  # Ajustar índice para que comience en 1
    st.dataframe(sector_counts)
    