# pages/2_Restauracion.py
import streamlit as st
import os
import subprocess
import time
from PIL import Image
import sys 

# --- Configuración de la página ---
st.set_page_config(page_title="Restaurador de QR", page_icon="🛠️", layout="wide")

# --- LÓGICA DE RUTAS INFALIBLES ---
PAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PAGE_DIR)
SQHN_DIR = os.path.join(ROOT_DIR, "SQHN")
CLEAN_FOLDER = os.path.join(SQHN_DIR, "qr_codes_clean")
CORRUPTED_FOLDER = os.path.join(SQHN_DIR, "qr_codes_corrupted")
PLOTS_FOLDER = os.path.join(SQHN_DIR, "plots")
LIBRARY_FILE = os.path.join(SQHN_DIR, "qr_model_library.pkl")
SCRIPT_PATH = os.path.join(SQHN_DIR, "qr_restorer_hires.py")

os.makedirs(CLEAN_FOLDER, exist_ok=True)
os.makedirs(CORRUPTED_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# --- INICIALIZACIÓN DE LA MEMORIA DE LA APP (st.session_state) ---
# Esto solo se ejecuta una vez al principio de la sesión.
if 'step' not in st.session_state:
    st.session_state.step = "1_upload_clean"
if 'library_trained' not in st.session_state:
    st.session_state.library_trained = False

def restart_process():
    """Función para reiniciar el flujo de trabajo."""
    st.session_state.step = "1_upload_clean"
    st.session_state.library_trained = False
    # Opcional: Borrar archivos antiguos para empezar completamente de cero
    if os.path.exists(LIBRARY_FILE):
        os.remove(LIBRARY_FILE)
    st.success("Proceso reiniciado. Puedes empezar desde el Paso 1.")
    time.sleep(2) # Pausa para que el usuario vea el mensaje

# --- Contenido de la Página ---
st.title("🛠️ Herramienta de Restauración de Códigos QR")
st.sidebar.button("Empezar de Nuevo", on_click=restart_process)
st.sidebar.markdown("---")
resolution = st.sidebar.number_input("Resolución (px)", min_value=32, max_value=256, value=96, step=32)
st.sidebar.markdown("---")

# --- PASO 1: Subir Imágenes Limpias ---
st.header("Paso 1: Sube los QR Limpios de Referencia")
if st.session_state.step == "1_upload_clean":
    uploaded_clean_files = st.file_uploader(
        "Selecciona los archivos de QR limpios que formarán tu 'biblioteca'",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="clean_uploader"
    )
    if uploaded_clean_files:
        for uploaded_file in uploaded_clean_files:
            with open(os.path.join(CLEAN_FOLDER, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"{len(uploaded_clean_files)} imágenes limpias cargadas.")
        st.session_state.step = "2_train" # Avanzamos al siguiente paso
        st.rerun() # Forzamos el reinicio para mostrar el siguiente paso

# --- PASO 2: Construir la Biblioteca de Modelos ---
if st.session_state.step == "2_train":
    st.header("Paso 2: Construye la Biblioteca de Modelos")
    st.info("Ahora que las imágenes están cargadas, haz clic en el botón para entrenar los modelos.")
    
    if st.button("Entrenar Modelos", type="primary"):
        with st.spinner(f"Entrenando modelos a {resolution}x{resolution}px... Esto puede tardar varios minutos."):
            if os.path.exists(LIBRARY_FILE): os.remove(LIBRARY_FILE)
            command = ["python", SCRIPT_PATH, "learn", CLEAN_FOLDER, "--resolution", str(resolution)]
            process = subprocess.run(command, capture_output=True, text=True)
            
            if process.returncode == 0:
                st.success("¡Biblioteca de modelos construida exitosamente!")
                st.session_state.library_trained = True
                st.session_state.step = "3_restore" # Avanzamos al paso final
                st.rerun()
            else:
                st.error("Ocurrió un error durante el entrenamiento.")
                st.expander("Ver detalles del error").code(process.stderr)

# --- PASO 3: Subir Imagen Corrupta y Restaurar ---
if st.session_state.step == "3_restore":
    st.header("Paso 3: Sube un QR Corrupto y Restáuralo")
    if not st.session_state.library_trained:
        st.warning("Primero debes entrenar los modelos en el Paso 2.")
    else:
        uploaded_corrupted_file = st.file_uploader(
            "Selecciona el archivo de QR dañado que quieres restaurar",
            type=["png", "jpg", "jpeg"],
            key="corrupted_uploader"
        )
        if uploaded_corrupted_file:
            corrupted_path = os.path.join(CORRUPTED_FOLDER, uploaded_corrupted_file.name)
            with open(corrupted_path, "wb") as f:
                f.write(uploaded_corrupted_file.getbuffer())

            with st.spinner("Restaurando la imagen..."):
                command = ["python", SCRIPT_PATH, "restore", corrupted_path, "--resolution", str(resolution)]
                process = subprocess.run(command, capture_output=True, text=True)

                if process.returncode == 0:
                    st.success("¡Imagen restaurada con éxito!")
                    output_filename = f"qr_restoration_{resolution}px_result.png"
                    result_image_path = os.path.join(PLOTS_FOLDER, output_filename)
                    if os.path.exists(result_image_path):
                        image = Image.open(result_image_path)
                        st.image(image, caption="Resultado Final de la Restauración", use_column_width=True)
                    else:
                        st.error("El archivo de imagen del resultado no fue encontrado.")
                else:
                    st.error("Ocurrió un error durante la restauración.")
                    st.expander("Ver detalles del error").code(process.stderr)

# Mostrar un mensaje de estado si el proceso ya se completó
if st.session_state.step == "3_restore" and st.session_state.library_trained:
    st.success("La biblioteca de modelos ya está entrenada. Puedes subir un archivo corrupto para restaurarlo.")
