import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Conócenos", layout="wide")

# Contenedor centrado para "Conócenos" y su texto
col_intro1, col_intro2, col_intro3 = st.columns([1, 2, 1])  # Márgenes laterales
with col_intro2:
    st.markdown(
        "# Conócenos"  # Cambiado de ## a # para hacerlo más grande
    )
    st.markdown(
        """
        <div style="text-align: justify;">
        Somos un equipo  de Business Analytics apasionado por la tecnología, la innovación y el desarrollo sostenible. 
        Nuestro objetivo es crear soluciones que impulsen el crecimiento económico y fomentar 
        un futuro más sostenible para todos.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")  # Espacio antes de la información de los integrantes

# Organización en dos filas con márgenes entre columnasre columnas
fila1_col1, margen1, fila1_col2, margen2 = st.columns([1, 0.2, 1, 0.2])
fila2_col1, margen3, fila2_col2, margen4 = st.columns([1, 0.2, 1, 0.2])
# Lista de integrantes con sus perfiles de LinkedIn
integrantes = [
    ("Valentina Bailón", "images/valen.png",
     """
     <div style="text-align: justify;"> 
     Me apasionan el análisis de datos, la estrategia empresarial y los retos intelectuales.Hablo español, inglés y tengo conocimientos básicos de francés e italiano.  
     Me considero una persona proactiva, curiosa y con ganas de aprender.
     </div>
     """,
     "https://www.linkedin.com/in/valentina-bailon-2653b22b7?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app"),
    ("Paulina Perales", "images/PAU.png",
     """
     <div style="text-align: justify;">
     Tengo un gran interés en la estadística 
     y el análisis de datos para la toma de decisiones estratégicas. Actualmente soy Secretaria General 
     de Comillas Filosofía e Historia, un club que fomenta el debate y la reflexión en el ambiente universitario.
     </div>
     """,
     "https://www.linkedin.com/in/paulinaperalescai%C3%B1a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app"),
    ("Jacobo Benlloch", "images/JACOB.png",
     """
     <div style="text-align: justify;">
     Soy desarrollador con experiencia en la creación de software, automatizaciones y soluciones digitales para pequeñas empresas. Me apasiona el desarrollo de programas, 
     la programación web y el mundo informático en general. Disfruto resolviendo problemas mediante la tecnología 
     y optimizando procesos a través de herramientas digitales.
     </div>
     """,
     "https://www.linkedin.com/in/jacobo-benlloch-70564a2a7?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app"),
    ("Lucía Alvira", "images/LUCIA.png",
     """
     <div style="text-align: justify;">
     Tengo pasión por extraer valor de los datos y convertir información en soluciones prácticas. Me motiva la resolución de problemas a través del pensamiento analítico y la optimización de procesos. Destaco por mi capacidad de adaptación y enfoque proactivo en entornos dinámicos.
     </div>
     """,
     "https://www.linkedin.com/in/luc%C3%ADa-alvira-mart%C3%ADn-8488a834a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app")
]

# Función para mostrar cada integrante con LinkedIn
def mostrar_integrante(col, nombre, imagen, descripcion, linkedin):
    with col:
        st.image(imagen, width=200)
        col1, col2 = st.columns([4, 1])  # División para el nombre y el logo
        with col1:
            st.subheader(nombre)
        with col2:
            # Usar una imagen de LinkedIn en lugar del emoji
            st.markdown(f'<a href="{linkedin}" target="_blank"><img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width="30"></a>', unsafe_allow_html=True)
        st.markdown(descripcion, unsafe_allow_html=True)

# Primera fila de integrantes
mostrar_integrante(fila1_col1, *integrantes[0])
mostrar_integrante(fila1_col2, *integrantes[1])
st.write("")  # Espacio entre filas

# Segunda fila de integrantes
mostrar_integrante(fila2_col1, *integrantes[2])
mostrar_integrante(fila2_col2, *integrantes[3])