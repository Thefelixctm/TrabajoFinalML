# pages/comparacion.py
import streamlit as st
import os
from PIL import Image

# --- Configuración específica de la página (opcional, pero buena práctica) ---
# Si bien st.set_page_config es global, a veces es útil para títulos de página
# que aparecen en la barra lateral en aplicaciones multipágina.
st.set_page_config(
    page_title="Gráficos de Comparación", # Título para esta página específica en la barra lateral
    page_icon="📊",
)

# --- Contenido de la página de Gráficos de Comparación ---
st.title("📊 Gráficos de Comparación de Rendimiento")
st.markdown("""
En esta sección, se presentan los resultados cuantitativos de nuestros experimentos,
comparando el rendimiento de los diferentes modelos en tareas de reconstrucción de imágenes.
""")

# --- RUTA A LA CARPETA DE GRÁFICOS ---
# NOTA IMPORTANTE: La ruta 'Graficos/' es relativa al directorio **raíz**
# desde donde ejecutas 'streamlit run inicio.py'.
# Asegúrate de que la carpeta 'Graficos/' esté en el mismo nivel que tu 'inicio.py'.
graficos_folder = "Graficos/"

# --- LISTA DE TUS GRÁFICOS Y SUS TÍTULOS/DESCRIPCIONES ---
# Puedes ajustar los títulos y descripciones según lo que represente cada gráfico
charts_to_display = [
    {"file": "OnCont_L1_Continual.png", "title": "Rendimiento Continual (Métrica L1)", "description": "Este gráfico muestra la evolución del rendimiento de los modelos en un escenario de aprendizaje continuo, utilizando la métrica L1. Permite observar cómo se adaptan a la información nueva y cómo retienen la aprendida."},
    {"file": "OnCont_L1_Cumulative.png", "title": "Rendimiento Acumulativo (Métrica L1)", "description": "Aquí se visualiza el rendimiento acumulativo de los modelos a lo largo de las fases de entrenamiento o evaluación, basado en la métrica L1. Esto indica la capacidad general de los modelos a medida que procesan más datos."},
    {"file": "OnCont_L1_Final_Performance.png", "title": "Rendimiento Final Global (Métrica L1)", "description": "Este gráfico resume el rendimiento final de cada modelo una vez completados todos los procesos de entrenamiento y evaluación principales, usando la métrica L1. Ofrece una comparación directa de su efectividad general."},
    {"file": "OnCont_L1_Sensitivity.png", "title": "Análisis de Sensibilidad (Métrica L1)", "description": "Explora cómo el rendimiento de los modelos, medido por L1, varía frente a cambios en parámetros clave o condiciones de entrada. Ayuda a comprender la robustez y las limitaciones de cada enfoque."}
]

# --- Cargar y mostrar los gráficos ---
if os.path.exists(graficos_folder) and os.path.isdir(graficos_folder):
    for chart_info in charts_to_display:
        chart_path = os.path.join(graficos_folder, chart_info["file"])
        if os.path.exists(chart_path):
            st.subheader(chart_info["title"])
            st.image(chart_path, caption=chart_info["description"], use_container_width=True)
            st.markdown("---") # Un separador visual para cada gráfico
        else:
            st.warning(f"No se encontró el gráfico: **{chart_info['file']}** en la ruta esperada.")
            st.markdown(f"Por favor, verifica que el archivo `{chart_info['file']}` esté dentro de la carpeta `{graficos_folder}` y que la carpeta `{graficos_folder}` esté en el mismo directorio que tu `inicio.py`.")
else:
    st.error(f"La carpeta de gráficos '{graficos_folder}' no se encontró. Asegúrate de que exista y esté en el mismo nivel que tu archivo 'inicio.py'.")
