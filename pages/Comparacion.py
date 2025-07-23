# pages/comparacion.py 
import streamlit as st
import os
from PIL import Image
import subprocess
import time

# --- Configuración de la página ---
st.set_page_config(
    page_title="Gráficos y Demos",
    page_icon="📊",
    layout="wide"
)

# --- LÓGICA DE RUTAS INFALIBLES ---
PAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PAGE_DIR)
SQHN_DIR = os.path.join(ROOT_DIR, "SQHN")
MAIN_SCRIPT_PATH = os.path.join(SQHN_DIR, "main.py")
PLOTS_OUTPUT_FOLDER = os.path.join(SQHN_DIR, "plots")
RESULTS_FOLDER = os.path.join(SQHN_DIR, "results")
# <-- NUEVO: Ruta a la carpeta de recursos estáticos
ASSETS_FOLDER = os.path.join(SQHN_DIR, "assets")

# Asegurarse de que la carpeta de salida para los gráficos exista
os.makedirs(PLOTS_OUTPUT_FOLDER, exist_ok=True)

# --- HERRAMIENTA DE DEPURACIÓN DE ARCHIVOS ---
with st.expander("🔍 Haz clic aquí para ver la estructura de archivos del servidor"):
    st.write("Directorio Raíz del Proyecto:")
    # Muestra el contenido del directorio raíz
    try:
        root_content = os.listdir(os.path.dirname(os.path.abspath(__file__)))
        st.code('\n'.join(root_content))
    except Exception as e:
        st.error(f"No se pudo leer el directorio raíz: {e}")
        
    st.write("Contenido de la carpeta SQHN/results/:")
    # Muestra el contenido de la carpeta de resultados
    results_path_debug = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SQHN', 'results')
    try:
        results_content = os.listdir(results_path_debug)
        if results_content:
            st.code('\n'.join(results_content))
        else:
            st.warning("¡La carpeta 'results' está vacía!")
    except FileNotFoundError:
        st.error("¡ERROR CRÍTICO! La carpeta 'SQHN/results' no existe en el servidor.")
    except Exception as e:
        st.error(f"No se pudo leer la carpeta 'results': {e}")

# --- Contenido de la Página ---
st.title("📊 Panel de Visualización de Resultados")
st.markdown("---")

# --- Sección 1: Gráficos del Experimento Principal ---
st.header("1. Gráficos de Rendimiento del Experimento `OnCont-L1`")
st.markdown("""
    Haz clic en el botón para ejecutar el script `plot.py` y generar los gráficos
    comparativos del experimento principal. Este proceso lee los archivos `.data`
    pre-calculados de la carpeta `SQHN/results/`.
    """)

# Lista de los gráficos del experimento principal
expected_charts = [
    {"file": "OnCont_L1_Continual.png", "title": "Rendimiento Continual (Métrica L1)", "description": "Muestra la evolución del rendimiento en un escenario de aprendizaje continuo."},
    {"file": "OnCont_L1_Final_Performance.png", "title": "Rendimiento Final Global (Métrica L1)", "description": "Resume el rendimiento final de cada modelo una vez completados los procesos."},
    {"file": "OnCont_L1_Sensitivity.png", "title": "Análisis de Sensibilidad (Métrica L1)", "description": "Explora cómo el rendimiento varía frente a cambios en parámetros clave."}
]


if st.button("Generar Gráficos del Experimento Principal"):
    if not os.path.exists(RESULTS_FOLDER) or not os.listdir(RESULTS_FOLDER):
        st.error("Error: La carpeta 'SQHN/results' está vacía.")
    else:
        with st.spinner("Generando los tres gráficos..."):
            try:
                process = subprocess.run(
                    ["python", "main.py", "--plot", "OnCont-L1"],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=SQHN_DIR 
                )

                st.success("¡Proceso de generación de gráficos completado!")
                time.sleep(1)

                # --- LÓGICA DE VISUALIZACIÓN MEJORADA CON TÍTULOS Y DESCRIPCIONES ---
                st.subheader("Resultados del Experimento Principal")
                
                # Mostramos cada gráfico con su título y descripción
                for chart in expected_charts:
                    chart_path = os.path.join(PLOTS_OUTPUT_FOLDER, chart["file"])
                    if os.path.exists(chart_path):
                        st.markdown(f"#### {chart['title']}") # <-- Usamos un sub-subtítulo para el título
                        image = Image.open(chart_path)
                        # <-- Usamos el parámetro 'caption' para la descripción
                        st.image(image, caption=chart["description"], use_column_width=True) 
                        st.markdown("---")
                    else:
                        st.warning(f"No se encontró el archivo del gráfico: {chart['file']}")

            except subprocess.CalledProcessError as e:
                st.error("Error al ejecutar el script de ploteo.")
                st.expander("Ver detalles del error").code(e.stderr, language='bash')
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")

st.markdown("---")

# --- Sección 2: Demostración Visual de Memoria "One-Shot" ---
st.header("2. Demostración Visual: Memoria 'One-Shot' (Fortaleza del SQHN)")

# --- LÓGICA DE BARRA DESPLEGABLE (st.expander) ---
with st.expander("Haz clic aquí para ver la Demostración Visual"):
    st.markdown("""
        Esta imagen demuestra la capacidad del **SQHN** para actuar como una **memoria perfecta**.
        Para cada dígito, se creó un modelo SQHN nuevo y se le enseñó esa única imagen.
        El resultado es una reconstrucción nítida y de alto contraste, a diferencia del Autoencoder
        que, al haber sido entrenado con todas las imágenes, produce una versión más generalista y suave.
        """)

    oneshot_demo_image_path = os.path.join(ASSETS_FOLDER, "demo_oneshot.png")

    if os.path.exists(oneshot_demo_image_path):
        image = Image.open(oneshot_demo_image_path)
        # --- LÓGICA DE TAMAÑO CONTROLADO (width=...) ---
        st.image(image, caption="Columna 1: Original, Columna 2: SQHN (Memoria Perfecta), Columna 3: Autoencoder (Generalista)", width=700)
    else:
        st.error("¡Archivo de imagen de demostración no encontrado!")
        st.warning(f"Asegúrate de que la imagen 'demo_oneshot.png' exista en la carpeta: {ASSETS_FOLDER}")
