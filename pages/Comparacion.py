# pages/comparacion.py 
import streamlit as st
import os
from PIL import Image
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt

# --- Configuración específica de la página ---
st.set_page_config(
    page_title="Gráficos de Comparación",
    page_icon="📊",
)

# --- Contenido de la página de Gráficos de Comparación ---
st.title("📊 Generar y Ver Gráficos de Rendimiento")
st.markdown("""
    Haz clic en el botón de abajo para ejecutar el script de generación de gráficos.
    Este proceso creará los archivos de imagen de los gráficos en una carpeta temporal local
    (`SQHN/plots/`) y luego los mostrará directamente aquí.
    """)

# --- Configuración de rutas (VERIFICA ESTAS RUTAS) ---
# La ruta a tu script main.py (asumiendo que está en una subcarpeta 'SQHN')
main_script_path = "SQHN/main.py"
# La carpeta donde main.py guardará los gráficos generados (relativa al directorio raíz de tu app.py/inicio.py)
plots_output_folder = "SQHN/plots"

# Asegúrate de que la carpeta de salida de los gráficos exista
os.makedirs(plots_output_folder, exist_ok=True)


# --- Lista de los gráficos esperados ---
expected_charts = [
    {"file": "OnCont_L1_Continual.png", "title": "Rendimiento Continual (Métrica L1)", "description": "Muestra la evolución del rendimiento en un escenario de aprendizaje continuo."},
    {"file": "OnCont_L1_Cumulative.png", "title": "Rendimiento Acumulativo (Métrica L1)", "description": "Visualiza el rendimiento acumulativo de los modelos a lo largo de las fases."},
    {"file": "OnCont_L1_Final_Performance.png", "title": "Rendimiento Final Global (Métrica L1)", "description": "Resume el rendimiento final de cada modelo una vez completados los procesos."},
    {"file": "OnCont_L1_Sensitivity.png", "title": "Análisis de Sensibilidad (Métrica L1)", "description": "Explora cómo el rendimiento varía frente a cambios en parámetros clave."}
]


if st.button("Generar y Mostrar Gráficos"):
    st.info("Generando gráficos... Esto puede tardar un momento.")

    try:
        # Ejecutar el script main.py para generar los gráficos
        process = subprocess.run(
            ["python", main_script_path, "--plot", "OnCont-L1"],
            capture_output=True,
            text=True,
            check=True,
            # Se ELIMINA el argumento cwd para que el comando se ejecute desde la raíz del proyecto.
            # Así, 'SQHN/main.py' se resuelve correctamente desde 'Trabajo Final/'.
        )

        st.success("¡Gráficos generados exitosamente!")
        # Puedes descomentar estas líneas para ver la salida/errores del script main.py si hay problemas:
        # if process.stdout:
        #     st.expander("Ver salida del script (stdout)").code(process.stdout, language='bash')
        # if process.stderr:
        #     st.expander("Ver errores del script (stderr)").code(process.stderr, language='bash')

        # Pequeña pausa para asegurar que los archivos se hayan escrito completamente en el disco
        time.sleep(1)

        # Mostrar los gráficos generados
        for chart_info in expected_charts:
            chart_path = os.path.join(plots_output_folder, chart_info["file"])
            if os.path.exists(chart_path):
                st.subheader(chart_info["title"])
                st.image(chart_path, caption=chart_info["description"], use_column_width=True)
                st.markdown("---")
            else:
                st.warning(f"El archivo del gráfico no se encontró después de la generación: **{chart_info['file']}**")
                st.markdown(f"Verifica que el script `{main_script_path}` esté guardando los gráficos en la ruta correcta (`{plots_output_folder}`).")

    except subprocess.CalledProcessError as e:
        st.error(f"Error al ejecutar el script de generación de gráficos. Código de salida: {e.returncode}")
        st.code(e.stdout, language='bash')
        st.code(e.stderr, language='bash')
        st.warning(f"Asegúrate de que `{main_script_path}` exista, sea ejecutable y que todas sus dependencias estén instaladas.")
    except FileNotFoundError:
        st.error(f"El comando 'python' o el script '{main_script_path}' no se encontró.")
        st.warning("Asegúrate de que Python esté en tu PATH y que la ruta al script `main.py` sea correcta.")
    except Exception as e:
        st.error(f"Ocurrió un error inesperado durante la generación o visualización: {e}")

else:
    st.info("Haz clic en el botón para comenzar la generación de gráficos.")
