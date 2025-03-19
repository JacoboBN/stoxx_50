import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Nuestra Selección - Top 5 Acciones STOXX50", layout="wide")
st.title("Nuestra Selección - Top 5 Acciones STOXX50")

# Variables globales
DEBUG_MODE = False

# Definición de métricas por categoría con sus respectivos pesos
METRICS = {
    'growth': {
        'metrics': ['Revenue', 'Sales and Services Revenues', 'Diluted EPS from Continuing Operations - Adjusted'],
        'weight': 0.125  # Reducido para dar espacio a ESG
    },
    'profitability': {
        'metrics': ['Return on Equity', 'EBITDA Margin Adjusted', 'Free Cash Flow'],
        'weight': 0.20  # Reducido para dar espacio a ESG
    },
    'valuation': {
        'metrics': ['Price Earnings Ratio (P/E)', 'EBITDA Adjusted'],
        'weight': 0.125  # Reducido para dar espacio a ESG
    },
    'leverage': {
        'metrics': ['WACC Total Invested Capital', 'WACC Total Capital'],
        'weight': 0.05
    },
    'esg': {
        'metrics': ['BESG ESG Score', 'BESG Environmental Pillar Score', 'BESG Social Pillar Score', 
                    'BESG Governance Pillar Score', 'BESG Percent Board Members that are Women Fld Scr'],
        'weight': 0.50  # Nuevo peso para la categoría ESG
    }
}

# Métricas donde valores más bajos son mejores
INVERSE_METRICS = ['Price Earnings Ratio (P/E)', 'WACC Total Invested Capital', 'WACC Total Capital']

def load_data():
    try:
        # Cargar datos financieros principales
        df = pd.read_excel("Datos_STOXX50_.xlsx", header=[0, 1])
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Cargar datos ESG desde la hoja específica
        df_esg = pd.read_excel("Datos_STOXX50_.xlsx", sheet_name="ESG", header=[0, 1])
        esg_date_col = df_esg.columns[0]
        df_esg[esg_date_col] = pd.to_datetime(df_esg[esg_date_col])
        
        # Cargar datos de mapeo de nombres desde la hoja Sector
        df_sector = pd.read_excel("Datos_STOXX50_.xlsx", sheet_name="Sector")
        
        # Crear diccionario de mapeo de nombres, sectores y países
        name_mapping = dict(zip(df_sector.iloc[:, 0], df_sector.iloc[:, 1]))
        sector_mapping = dict(zip(df_sector.iloc[:, 0], df_sector.iloc[:, 2]))
        country_mapping = dict(zip(df_sector.iloc[:, 0], df_sector.iloc[:, 3]))
        
        return df.set_index(date_col), df_esg.set_index(esg_date_col), name_mapping, sector_mapping, country_mapping, df_sector
    except FileNotFoundError:
        st.error("Error: No se pudo encontrar el archivo 'Datos_STOXX50_.xlsx'.")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")
        if DEBUG_MODE:
            import traceback
            st.code(traceback.format_exc())
        return None, None, None, None, None, None

def extract_years(dates):
    years = []
    
    if isinstance(dates, pd.DatetimeIndex):
        return dates.year.tolist()
    
    for date in dates:
        year = None
        if isinstance(date, (pd.Timestamp, datetime)):
            year = date.year
        elif isinstance(date, str) and re.match(r'\d{4}-\d{2}-\d{2}', date):
            year = int(date.split('-')[0])
        elif isinstance(date, str):
            year_match = re.search(r'(\d{4})', date)
            if year_match:
                potential_year = int(year_match.group(1))
                year = potential_year if 1900 <= potential_year <= 2100 else None
        elif isinstance(date, (int, float)) and not pd.isna(date) and 1900 <= date <= 2100:
            year = int(date)
        
        years.append(year)
    
    return years

def extract_metrics_by_action(df, years):
    actions_metrics = {}
    metrics_by_action = {}
    
    for col in df.columns:
        if pd.isna(col[0]) or pd.isna(col[1]) or 'Dates' in str(col[1]) or 'date' in str(col[1]).lower() or 'Unnamed' in str(col[0]):
            continue
        
        ticker = col[0]
        metric_name = col[1]
        base_ticker = re.sub(r'\.\d+$', '', ticker)
        
        if base_ticker not in actions_metrics:
            actions_metrics[base_ticker] = []
            metrics_by_action[base_ticker] = {}
            
        actions_metrics[base_ticker].append(metric_name)
        metrics_by_action[base_ticker][metric_name] = df[col].values
    
    return actions_metrics, metrics_by_action

def categorize_metrics(action, metrics, years):
    action_debug_info = {
        'metrics_found': [],
        'years_with_data': {},
        'valid_categories': 0
    }
    
    categorized_metrics = {category: {} for category in METRICS.keys()}
    
    for metric_name, values in metrics.items():
        for category, category_data in METRICS.items():
            for ref_metric in category_data['metrics']:
                # Comparación flexible para encontrar coincidencias
                if (ref_metric.lower() in metric_name.lower() or 
                    metric_name.lower() in ref_metric.lower() or
                    any(word.lower() in metric_name.lower() for word in ref_metric.lower().split())):
                    
                    yearly_data = {}
                    
                    for i, (year, value) in enumerate(zip(years, values)):
                        if year is not None and not pd.isna(value) and isinstance(value, (int, float)):
                            yearly_data[year] = value
                            
                            if year not in action_debug_info['years_with_data']:
                                action_debug_info['years_with_data'][year] = 0
                            action_debug_info['years_with_data'][year] += 1
                    
                    if yearly_data:
                        categorized_metrics[category][ref_metric] = yearly_data
                        if ref_metric not in action_debug_info['metrics_found']:
                            action_debug_info['metrics_found'].append(ref_metric)
                        break
    
    # Contar categorías válidas
    action_debug_info['valid_categories'] = sum(1 for category_metrics in categorized_metrics.values() if category_metrics)
    
    return categorized_metrics, action_debug_info

def collect_all_metric_values(metrics_by_action):
    all_metric_values = {}
    
    for metrics_dict in metrics_by_action.values():
        for metric_name, values in metrics_dict.items():
            for category, category_data in METRICS.items():
                for ref_metric in category_data['metrics']:
                    if (ref_metric.lower() in metric_name.lower() or 
                        metric_name.lower() in ref_metric.lower() or
                        any(word.lower() in metric_name.lower() for word in ref_metric.lower().split())):
                        
                        if ref_metric not in all_metric_values:
                            all_metric_values[ref_metric] = []
                        
                        all_metric_values[ref_metric].extend([v for v in values if isinstance(v, (int, float)) and not pd.isna(v)])
    
    return all_metric_values

def calculate_scores(action, categorized_metrics, all_metric_values, all_years):
    if not all_years:
        return 0, {}, 0
    
    # Calcular ponderación temporal (años más recientes tienen más peso)
    min_year = min(all_years)
    time_weights = {year: 1.5 ** (year - min_year) for year in all_years}
    total_time_weight = sum(time_weights.values())
    normalized_time_weights = {year: weight / total_time_weight for year, weight in time_weights.items()}
    
    # Calcular puntuaciones por categoría
    category_scores = {category: 0 for category in categorized_metrics.keys()}
    category_counts = {category: 0 for category in categorized_metrics.keys()}
    
    # Calcular puntuaciones normalizadas por categoría
    for category, metrics_data in categorized_metrics.items():
        for metric, yearly_data in metrics_data.items():
            if metric not in all_metric_values or len(all_metric_values[metric]) < 2:
                continue
            
            metric_values = all_metric_values[metric]
            min_val, max_val = min(metric_values), max(metric_values)
            
            # Normalizar valores
            normalized_values = {}
            for y, value in yearly_data.items():
                try:
                    if max_val == min_val:
                        norm_value = 0.5
                    else:
                        norm_value = 1 - (value - min_val) / (max_val - min_val) if metric in INVERSE_METRICS else (value - min_val) / (max_val - min_val)
                    
                    normalized_values[y] = norm_value
                except Exception as e:
                    if DEBUG_MODE:
                        st.warning(f"Error al normalizar {action} - {metric} - {y}: {e}")
            
            # Calcular puntuación ponderada por tiempo para esta métrica
            metric_score = sum(normalized_values.get(y, 0) * normalized_time_weights.get(y, 0) for y in all_years)
            
            if not pd.isna(metric_score):
                category_scores[category] += metric_score
                category_counts[category] += 1
    
    # Calcular puntuación final ponderando por categoría
    final_score = 0
    valid_categories = 0
    category_results = {}
    
    for category, category_data in METRICS.items():
        if category_counts[category] > 0:
            category_avg_score = category_scores[category] / category_counts[category]
            final_score += category_avg_score * category_data['weight']
            valid_categories += 1
            category_results[category] = category_avg_score
    
    return final_score, category_results, valid_categories

def merge_metrics_data(financial_data, esg_data, financial_years, esg_years):
    """
    Combina los datos financieros y ESG para todas las acciones
    """
    merged_actions_metrics = {}
    merged_metrics_by_action = {}
    
    # Primero agregamos los datos financieros
    for action, metrics in financial_data.items():
        if action not in merged_metrics_by_action:
            merged_metrics_by_action[action] = {}
            merged_actions_metrics[action] = []
        
        for metric_name, values in metrics.items():
            merged_metrics_by_action[action][metric_name] = values
            merged_actions_metrics[action].append(metric_name)
    
    # Luego agregamos los datos ESG
    for action, metrics in esg_data.items():
        if action not in merged_metrics_by_action:
            merged_metrics_by_action[action] = {}
            merged_actions_metrics[action] = []
        
        for metric_name, values in metrics.items():
            merged_metrics_by_action[action][metric_name] = values
            merged_actions_metrics[action].append(metric_name)
    
    return merged_actions_metrics, merged_metrics_by_action

def create_distribution_charts(all_actions, top_actions, sector_mapping, country_mapping, name_mapping):
    """
    Crea gráficos de distribución interactivos por país y sector utilizando Plotly
    """
    # Preparar datos para gráficos
    all_countries = [country_mapping.get(action, "Desconocido") for action in all_actions]
    all_sectors = [sector_mapping.get(action, "Desconocido") for action in all_actions]
    
    # Preparar datos para los top 5 con sus nombres completos
    top_countries_data = [(action, name_mapping.get(action, action), country_mapping.get(action, "Desconocido")) 
                          for action in top_actions[:5]]  # Solo considerar las 5 mejores acciones
    top_sectors_data = [(action, name_mapping.get(action, action), sector_mapping.get(action, "Desconocido")) 
                        for action in top_actions[:5]]  # Solo considerar las 5 mejores acciones
    
    # Crear dataframes
    all_countries_df = pd.DataFrame({'País': all_countries}).value_counts().reset_index()
    all_countries_df.columns = ['País', 'Cantidad']
    
    all_sectors_df = pd.DataFrame({'Sector': all_sectors}).value_counts().reset_index()
    all_sectors_df.columns = ['Sector', 'Cantidad']
    
    top_countries_df = pd.DataFrame([country for _, _, country in top_countries_data], columns=['País']).value_counts().reset_index()
    top_countries_df.columns = ['País', 'Cantidad']
    
    top_sectors_df = pd.DataFrame([sector for _, _, sector in top_sectors_data], columns=['Sector']).value_counts().reset_index()
    top_sectors_df.columns = ['Sector', 'Cantidad']
    
    # Crear figuras interactivas con Plotly
    # Para todas las acciones por país
    fig1 = px.bar(
        all_countries_df,
        x='País',
        y='Cantidad',
        title='Distribución por Países - Todas las Acciones',
        color='Cantidad',
        color_continuous_scale='Blues',
        labels={'Cantidad': 'Número de acciones'},
        height=500
    )
    fig1.update_layout(xaxis_title='País', yaxis_title='Cantidad de Acciones')
    
    # Para top 5 por país
    fig2 = px.pie(
        top_countries_df,
        names='País',
        values='Cantidad',
        title='Distribución por Países - Top 5 Acciones',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    
    # Para todas las acciones por sector
    fig3 = px.bar(
        all_sectors_df,
        x='Sector',
        y='Cantidad',
        title='Distribución por Sectores - Todas las Acciones',
        color='Cantidad',
        color_continuous_scale='Reds',
        labels={'Cantidad': 'Número de acciones'},
        height=500
    )
    fig3.update_layout(xaxis_title='Sector', yaxis_title='Cantidad de Acciones')
    
    fig4 = px.pie(
        top_sectors_df,
        names='Sector',
        values='Cantidad',
        title='Distribución por Sectores - Top 5 Acciones',
        color_discrete_sequence=px.colors.qualitative.Set1  # Usar una secuencia de colores cualitativa
    )
    fig4.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont_size=14  # Aumentar el tamaño de la fuente para mejor legibilidad
    )
    # Añadir texto sobre las barras mostrando las acciones específicas
    for sector in top_sectors_df['Sector']:
        acciones_en_sector = [nombre for _, nombre, sec in top_sectors_data if sec == sector]
        acciones_texto = "<br>".join(acciones_en_sector)
        
        fig4.add_annotation(
            x=sector,
            y=top_sectors_df[top_sectors_df['Sector'] == sector]['Cantidad'].values[0],
            text=str(len(acciones_en_sector)),
            showarrow=False,
            yshift=10,
            font=dict(color='black', size=12)
        )
    
    # Personalizar hovers para mostrar qué acciones específicas están en cada categoría
    top_countries_hover = {}
    for action, name, country in top_countries_data:
        if country not in top_countries_hover:
            top_countries_hover[country] = []
        top_countries_hover[country].append(name)
    
    top_sectors_hover = {}
    for action, name, sector in top_sectors_data:
        if sector not in top_sectors_hover:
            top_sectors_hover[sector] = []
        top_sectors_hover[sector].append(name)
    
    # Actualizar hovertemplate
    fig2.update_traces(
        hovertemplate='<b>%{label}</b><br>' +
                      'Cantidad: %{value}<br>' +
                      'Acciones: %{customdata}<br>' +
                      '<extra></extra>',
        customdata=[', '.join(top_countries_hover[country]) for country in top_countries_df['País']]
    )
    
    fig4.update_traces(
        hovertemplate='<b>%{label}</b><br>' +
                      'Cantidad: %{value}<br>' +
                      'Acciones: %{customdata}<br>' +
                      '<extra></extra>',
        customdata=[', '.join(top_sectors_hover[sector]) for sector in top_sectors_df['Sector']]
    )
    
    return fig1, fig2, fig3, fig4, top_countries_hover, top_sectors_hover

def display_results(sorted_actions, name_mapping, sector_mapping, country_mapping, action_debug_info=None):
    if not sorted_actions:
        st.error("No se encontraron acciones con datos suficientes para calcular puntuaciones.")
        
        if DEBUG_MODE and action_debug_info:
            st.subheader("Diagnóstico")
            for action, info in action_debug_info.items():
                st.write(f"**{action}**:")
                st.write(f"- Métricas encontradas: {len(info['metrics_found'])}")
                st.write(f"- Métricas: {', '.join(info['metrics_found'])}")
                st.write(f"- Años con datos: {info['years_with_data']}")
                st.write(f"- Categorías válidas: {info['valid_categories']}")
        return
    
    # Aplicar filtros de diversificación
    filtered_actions = apply_diversity_filters(sorted_actions, sector_mapping, country_mapping, min_countries=3, min_sectors=3)
    
    # Mapear los nombres de acciones usando el diccionario de mapeo
    mapped_actions = []
    for action, score, category_scores in filtered_actions:
        # Usar el nombre mapeado si existe, o el original si no
        display_name = name_mapping.get(action, action)
        mapped_actions.append((action, display_name, score, category_scores))
    
    # Crear dataframe de resultados con nombres mapeados
    results_df = pd.DataFrame({
        "Acción": [display_name for _, display_name, _, _ in mapped_actions],
        "Puntuación": [round(score, 4) for _, _, score, _ in mapped_actions],
        "Ticker": [action for action, _, _, _ in mapped_actions],
        "Sector": [sector_mapping.get(action, "Desconocido") for action, _, _, _ in mapped_actions],
        "País": [country_mapping.get(action, "Desconocido") for action, _, _, _ in mapped_actions]
    })
    
    # Ajustar el índice para que empiece en 1
    results_df.index = results_df.index + 1
    
    # Mostrar información sobre diversificación
    unique_countries = results_df['País'].nunique()
    unique_sectors = results_df['Sector'].nunique()
    
    st.info(f"Diversificación: {unique_countries} países y {unique_sectors} sectores representados en las Top 5 acciones.")
    
    # Crear layout de una columna
    st.subheader("Top 5 Acciones Recomendadas")
    st.dataframe(results_df[["Acción", "Ticker", "Sector", "País", "Puntuación"]], use_container_width=True)

    st.subheader("Gráfico de Puntuaciones")

    # Crear gráfico interactivo con Plotly
    fig = px.bar(
        results_df,
        x="Acción",
        y="Puntuación",
        color="Puntuación",
        color_continuous_scale="Viridis",
        text="Puntuación",
        labels={"Puntuación": "Puntuación Total"},
        height=550
    )

    fig.update_traces(
        texttemplate='%{text:.3f}',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Puntuación: %{y:.4f}<br>Sector: %{customdata[0]}<br>País: %{customdata[1]}<extra></extra>',
        customdata=results_df[['Sector', 'País']].values
    )

    fig.update_layout(
        title="Top 5 Acciones por Puntuación",
        xaxis_title="Acción",
        yaxis_title="Puntuación",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar desglose por categorías
    st.subheader("Desglose por Categorías")
    
    category_translation = {
        'growth': 'Crecimiento',
        'profitability': 'Rentabilidad',
        'valuation': 'Valoración',
        'leverage': 'Apalancamiento',
        'esg': 'ESG'  # Nueva categoría
    }
    
    # Crear datos para gráfico radial
    categories_data = []
    for action, display_name, _, category_scores in mapped_actions:
        action_data = {"Acción": display_name, "Ticker": action}
        for category, translation in category_translation.items():
            action_data[translation] = round(category_scores.get(category, 0), 4)
        categories_data.append(action_data)
    
    categories_df = pd.DataFrame(categories_data)
    
    # Ajustar el índice para que empiece en 1
    categories_df.index = categories_df.index + 1
    
    # Mostrar tabla de categorías
    st.dataframe(categories_df, use_container_width=True)
    
    # Comparación directa de las 5 acciones en un solo gráfico
    st.subheader("Comparación de Todas las Acciones por Categoría")
    
    fig_comparison = go.Figure()
    
    for action, display_name, _, category_scores in mapped_actions[:5]:
        categories = [category_translation[cat] for cat in category_scores.keys()]
        values = [round(score, 3) for score in category_scores.values()]
        
        fig_comparison.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=display_name
        ))
    
    fig_comparison.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Comparación de Categorías entre las Top 5 Acciones",
        height=600
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Lista de todas las acciones y top 5 (para los gráficos)
    all_actions = list(action_debug_info.keys()) if action_debug_info else []
    top_actions = [action for action, _, _ in filtered_actions]
    
    # Crear y mostrar los gráficos de distribución interactivos
    fig1, fig2, fig3, fig4, top_countries_hover, top_sectors_hover = create_distribution_charts(
        all_actions, top_actions, sector_mapping, country_mapping, name_mapping
    )
    
    # Mostrar los gráficos en dos filas
    st.subheader("Análisis de Distribución")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        st.plotly_chart(fig4, use_container_width=True)
    
    # Mostrar leyenda sobre qué acciones específicas están en cada categoría
    st.subheader("Detalle de las Top 5 Acciones por País y Sector")
    
    col_detail1, col_detail2 = st.columns(2)
    
    with col_detail1:
        st.write("**Por País:**")
        for country, actions in top_countries_hover.items():
            st.write(f"**{country}**: {', '.join(actions)}")
    
    with col_detail2:
        st.write("**Por Sector:**")
        for sector, actions in top_sectors_hover.items():
            st.write(f"**{sector}**: {', '.join(actions)}")

def apply_diversity_filters(sorted_actions, sector_mapping, country_mapping, min_countries=3, min_sectors=3):

    selected_actions = []
    countries = set()
    sectors = set()
    
    # Comenzamos con todas las acciones ordenadas por puntuación
    remaining_actions = sorted_actions.copy()
    
    # Primera fase: seleccionamos la mejor acción para cada país hasta cumplir el mínimo
    while len(countries) < min_countries and remaining_actions:
        # Iteramos sobre las acciones restantes
        for i, (action, score, category_scores) in enumerate(remaining_actions):
            country = country_mapping.get(action, "Desconocido")
            
            # Si este país aún no está representado, seleccionamos esta acción
            if country not in countries:
                selected_actions.append((action, score, category_scores))
                countries.add(country)
                remaining_actions.pop(i)
                break
        else:
            # Si hemos revisado todas las acciones y no encontramos nuevos países, salimos del bucle
            break
    
    # Segunda fase: aseguramos el mínimo de sectores
    sectors = set(sector_mapping.get(action, "Desconocido") for action, _, _ in selected_actions)
    
    while len(sectors) < min_sectors and remaining_actions:
        for i, (action, score, category_scores) in enumerate(remaining_actions):
            sector = sector_mapping.get(action, "Desconocido")
            
            # Si este sector aún no está representado, seleccionamos esta acción
            if sector not in sectors:
                selected_actions.append((action, score, category_scores))
                sectors.add(sector)
                remaining_actions.pop(i)
                break
        else:
            # Si hemos revisado todas las acciones y no encontramos nuevos sectores, salimos del bucle
            break
    
    # Tercera fase: completar hasta 5 con las mejores acciones restantes
    while len(selected_actions) < 5 and remaining_actions:
        selected_actions.append(remaining_actions.pop(0))
    
    # Ordenamos nuevamente por puntuación para presentar en orden
    selected_actions.sort(key=lambda x: x[1], reverse=True)
    
    return selected_actions

def main():
    # Cargar datos financieros, ESG y el mapeo de nombres, sectores y países
    df_financial, df_esg, name_mapping, sector_mapping, country_mapping, df_sector = load_data()
    if df_financial is None or df_esg is None or name_mapping is None:
        return
    
    if DEBUG_MODE:
        st.subheader("Información de Estructura")
        st.write(f"Forma del DataFrame Financiero: {df_financial.shape}")
        st.write(f"Forma del DataFrame ESG: {df_esg.shape}")
        st.write(f"Muestra de columnas financieras: {list(df_financial.columns)[:5]}")
        st.write(f"Muestra de columnas ESG: {list(df_esg.columns)[:5]}")
        st.write(f"Mapeo de nombres: {name_mapping}")
        st.write(f"Mapeo de sectores: {sector_mapping}")
        st.write(f"Mapeo de países: {country_mapping}")
        
        with st.expander("Ver Datos Financieros Originales"):
            st.dataframe(df_financial)
            
        with st.expander("Ver Datos ESG Originales"):
            st.dataframe(df_esg)
            
        with st.expander("Ver Datos de Sector y País"):
            st.dataframe(df_sector)
    
    # Extraer años de ambos conjuntos de datos
    financial_years = extract_years(df_financial.index)
    esg_years = extract_years(df_esg.index)
    
    valid_financial_years = [y for y in financial_years if y is not None]
    valid_esg_years = [y for y in esg_years if y is not None]
    
    if not valid_financial_years or not valid_esg_years:
        st.error("No se pudieron identificar años en las columnas de fechas.")
        return
    
    if DEBUG_MODE:
        st.write(f"Años financieros identificados: {sorted(list(set(valid_financial_years)))}")
        st.write(f"Años ESG identificados: {sorted(list(set(valid_esg_years)))}")
    
    # Extraer métricas por acción para ambos conjuntos de datos
    financial_actions_metrics, financial_metrics_by_action = extract_metrics_by_action(df_financial, financial_years)
    esg_actions_metrics, esg_metrics_by_action = extract_metrics_by_action(df_esg, esg_years)
    
    # Combinar los datos financieros y ESG
    merged_actions_metrics, merged_metrics_by_action = merge_metrics_data(
        financial_metrics_by_action, esg_metrics_by_action, financial_years, esg_years
    )
    
    if DEBUG_MODE:
        st.subheader("Acciones y Métricas Identificadas (Combinadas)")
        for action, metrics in merged_actions_metrics.items():
            st.write(f"**{action}**: {len(metrics)} métricas")
            esg_metrics = [m for m in metrics if any(esg_m in m for esg_m in METRICS['esg']['metrics'])]
            if esg_metrics:
                st.write(f"Métricas ESG: {esg_metrics}")
    
    # Combinar todos los años
    all_years = list(set(valid_financial_years + valid_esg_years))
    action_scores = {}
    action_category_scores = {}
    action_debug_info = {}
    
    # Recopilar todos los valores de las métricas (combinados)
    all_metric_values = collect_all_metric_values(merged_metrics_by_action)
    
    # Categorizar métricas y recopilar años para cada acción
    for action, metrics in merged_metrics_by_action.items():
        if DEBUG_MODE:
            st.write(f"Procesando acción: {action}")
        
        # Utilizamos todos los años para la categorización
        all_years_for_action = financial_years + esg_years
        categorized_metrics, debug_info = categorize_metrics(action, metrics, all_years_for_action)
        action_debug_info[action] = debug_info
        
        action_years = []
        for metrics_data in categorized_metrics.values():
            for yearly_data in metrics_data.values():
                action_years.extend(yearly_data.keys())
        
        all_years.extend(set(action_years))
    
    all_years = sorted(set(all_years)) if all_years else []
    
    # Calcular puntuaciones finales
    for action, metrics in merged_metrics_by_action.items():
        all_years_for_action = financial_years + esg_years
        categorized_metrics, _ = categorize_metrics(action, metrics, all_years_for_action)
        final_score, category_scores, valid_categories = calculate_scores(
            action, categorized_metrics, all_metric_values, all_years
        )
        
        if valid_categories >= 1:
            action_scores[action] = final_score
            action_category_scores[action] = category_scores
            
            if DEBUG_MODE:
                st.write(f"Puntuación final para {action}: {final_score:.4f}")
                for category, score in category_scores.items():
                    st.write(f"- {category}: {score:.4f}")
        elif DEBUG_MODE:
            st.write(f"No hay suficientes categorías con datos para {action}")
    
    # Ordenar acciones por puntuación y mostrar top 5
    sorted_actions = [(action, score, action_category_scores.get(action, {})) 
                    for action, score in action_scores.items()]
    sorted_actions.sort(key=lambda x: x[1], reverse=True)
    
    display_results(sorted_actions, name_mapping, sector_mapping, country_mapping, action_debug_info)

if __name__ == "__main__":
    main()