import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Herramienta de Inversión EURO STOXX 50", layout="wide")

# Título principal
st.title("Herramienta de Inversión EURO STOXX 50")

# Sección: Objetivo de esta Herramienta
st.header("Objetivo de esta Herramienta")
st.markdown("""
Esta herramienta tiene como objetivo identificar las mejores oportunidades de inversión dentro del **EURO STOXX 50**, combinando factores financieros con criterios de sostenibilidad (ESG).

Para ello, se ha desarrollado un modelo de puntuación que integra métricas tradicionales de análisis financiero con evaluaciones de impacto ambiental, social y de gobernanza. Esto permite seleccionar empresas que no solo son rentables, sino también responsables y con menor riesgo a largo plazo.

El análisis considera aspectos como la estabilidad financiera, la capacidad de crecimiento, la rentabilidad, la valoración en el mercado y el nivel de endeudamiento, además de su impacto en la sociedad y el medio ambiente.
""")

# Sección: Metodología del Análisis
st.header("Metodología del Análisis: Justificación de Cada Elemento")
st.markdown("""
El sistema de selección sigue un enfoque estructurado y basado en datos cuantitativos, aplicando criterios objetivos para determinar qué acciones son las más adecuadas para invertir.

El proceso se divide en cuatro pasos clave:
""")

# Paso 1: Carga y Procesamiento de Datos
st.subheader("1. Carga y Procesamiento de Datos")
st.markdown("""
En esta fase, se importan datos financieros y de sostenibilidad (ESG) de fuentes externas, como Bloomberg, para todas las empresas del EURO STOXX 50. Estos datos se someten a un proceso de limpieza y normalización para garantizar su comparabilidad y precisión. Este paso es fundamental para asegurar que las métricas utilizadas en el análisis sean consistentes y confiables.
""")

# Paso 2: Cálculo de Puntuaciones y Ponderaciones
st.subheader("2. Cálculo de Puntuaciones y Ponderaciones")
st.markdown("""
En este paso, se otorga un peso del 50% a las métricas ESG y un 50% a las métricas **financieras, reflejando la creciente importancia de los factores de sostenibilidad en las** decisiones de inversión. Los datos más recientes reciben un mayor peso en el cálculo de la puntuación, ya que reflejan mejor el desempeño actual de cada empresa. A continuación, se detalla la justificación de cada categoría analizada:
""")

# Categorías de Análisis
st.markdown("""
- **Crecimiento (12.5%):** Esta categoría mide la capacidad de la empresa para expandirse y generar más ingresos a lo largo del tiempo. Se utilizan métricas como los ingresos totales y el crecimiento del beneficio por acción (EPS). El peso del 12.5% refleja que, aunque el crecimiento es importante, no todas las empresas en sectores consolidados tienen tasas de crecimiento altas.

- **Rentabilidad (20%):** Evalúa si la empresa genera beneficios de manera eficiente. Las métricas incluyen el retorno sobre el capital (ROE), el margen EBITDA ajustado y el flujo de caja libre. El peso del 20% se debe a que las empresas altamente rentables tienden a mantener mejor su valor a largo plazo.

- **Valoración (12.5%):** Analiza si la acción está sobrevalorada o infravalorada en el mercado mediante el ratio precio/beneficio (P/E) y el EV/EBITDA ajustado. El peso es del 12.5%, ya que la valoración, aunque importante, no siempre es determinante en el éxito a largo plazo.

- **Apalancamiento (5%):** Mide la dependencia de la empresa al endeudamiento, analizando el costo promedio del capital (WACC). El peso es del 5%, ya que un alto endeudamiento no siempre es un factor decisivo si la empresa genera suficiente flujo de caja.

- **ESG (50%):** Evalúa el impacto ambiental, social y de gobernanza mediante la puntuación ESG general, las puntuaciones específicas de medioambiente, sociedad y gobernanza, y el porcentaje de mujeres en el consejo de administración. El peso del 50% refleja que las empresas con mejor desempeño ESG tienden a atraer más inversores, reducir riesgos regulatorios y mantener una ventaja competitiva a largo plazo.
""")

# Paso 3: Aplicación de Restricciones para Diversificación
st.subheader("3. Aplicación de Restricciones para Diversificación")
st.markdown("""
Para evitar concentraciones excesivas, se establecen restricciones de diversificación:

- **Mínimo 3 países representados en la selección final, reduciendo el riesgo geográfico.**

- **Mínimo 3 sectores distintos, asegurando que la cartera no dependa de un único tipo** de industria.
""")

# Paso 4: Visualización y Selección de las 5 Mejores Acciones
st.subheader("4. Visualización y Selección de las 5 Mejores Acciones")
st.markdown("""
- **Gráficos interactivos:** Se presentan herramientas visuales para comparar empresas y explorar tendencias.

- **Selección final:** Se identifican las cinco acciones mejor calificadas según la metodología aplicada.
""")