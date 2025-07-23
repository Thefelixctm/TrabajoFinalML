import streamlit as st
import os # Aunque no lo usemos mucho ahora, es útil para el futuro

# --- Configuración de la página principal (General para toda la app) ---
st.set_page_config(
    page_title="Reconstrucción de Imágenes con SQHN y Autoencoder", # Título que aparece en la pestaña del navegador
    page_icon="🌌", # Ícono que aparece en la pestaña del navegador (puedes usar emojis)
    layout="wide" # Opciones: "centered" (por defecto) o "wide" (ocupa más ancho de la pantalla)
)

# --- Contenido de la Página de Inicio ---

# Título principal de la aplicación
st.title("🌌 Reconstrucción de Imágenes: SQHN, Autoencoders y Más allá")

# Introducción general
st.markdown("""
¡Bienvenido a nuestra aplicación interactiva dedicada a explorar el fascinante campo de la **reconstrucción de imágenes**!
Aquí podrás descubrir cómo diferentes modelos de Machine Learning abordan el desafío de restaurar
información visual degradada o incompleta.
""")

st.divider() # Una línea divisoria para mejorar la legibilidad

st.header("🔍 Contexto y Punto de Partida: El Sparse Quantized Hopfield Network (SQHN)")
st.markdown("""
Nuestro proyecto se inspira directamente en el trabajo de investigación presentado en el paper
"**A sparse quantized Hopfield network for online-continual memory**" (publicado en *Nature Communications*, 2024).
Este artículo explora las capacidades de las **Redes de Hopfield Cuantizadas Esparsas (SQHN)**
como un modelo neuromórfico avanzado para la memoria asociativa y la recuperación de patrones.

Las SQHN demuestran un gran potencial para almacenar y recuperar una vasta cantidad de patrones
con alta fidelidad, incluso cuando la entrada es ruidosa o parcial.
En el contexto de la reconstrucción de imágenes, esto significa que un SQHN puede "recordar"
una imagen completa a partir de una versión degradada o con información faltante.
""")
# Puedes añadir un enlace al paper si lo tienes accesible públicamente:
# st.markdown("[Lee el paper original aquí](https://www.nature.com/articles/s41467-024-46976-4)") # Reemplaza con el enlace real

st.header("🚀 Nuestra Innovación: La Fusión con Autoencoders")
st.markdown("""
Aunque las SQHN son potentes, buscamos llevar la calidad de la reconstrucción al siguiente nivel.
Nuestra principal contribución a este trabajo es la **integración y comparación con Autoencoders**.

Un **Autoencoder** es una red neuronal no supervisada diseñada para aprender una representación
eficiente y de baja dimensionalidad (codificación) de los datos de entrada, y luego reconstruirlos
a partir de esta representación. Su fuerza radica en su capacidad para aprender características
esenciales de las imágenes y generar reconstrucciones con gran detalle y fidelidad,
incluso en escenarios complejos.

**Nuestra hipótesis central es que al combinar la robusta capacidad de memoria asociativa
de las SQHN con la habilidad de los Autoencoders para aprender representaciones latentes ricas
y generar reconstrucciones detalladas, podemos lograr una calidad de reconstrucción
significativamente superior.** Esto es especialmente relevante para capturar texturas finas
y estructuras complejas que podrían ser un desafío para modelos que dependen
exclusivamente de la recuperación de patrones discretos.
""")

st.header("🎯 Objetivo de Esta Aplicación")
st.markdown("""
Esta herramienta interactiva ha sido diseñada para que puedas:
1.  **Comprender** el concepto de reconstrucción de imágenes y las metodologías empleadas.
2.  **Visualizar** de forma clara y directa las diferencias en el rendimiento entre:
    * Los modelos basados puramente en **SQHN**.
    * Nuestra implementación de **Autoencoders**.
    * Posibles **combinaciones o mejoras** que hemos explorado (ej., un Autoencoder que preprocesa la imagen para el SQHN, o un SQHN que refina la salida de un Autoencoder).
3.  **Analizar** métricas cuantitativas y **comparar** visualmente las reconstrucciones.
""")

st.divider()

st.info("""
**Para empezar:** Utiliza el menú lateral (sidebar) para navegar entre las diferentes secciones:
* **Gráficos de Comparación:** Donde podrás ver el análisis cuantitativo del rendimiento.
* **Reconstrucción de Imágenes:** Donde podrás interactuar con los modelos y ver las reconstrucciones en tiempo real.
""")

# Opcional: Añadir una imagen o GIF representativo de la reconstrucción
# st.image("ruta/a/tu/imagen_reconstruccion.png", caption="Ejemplo de reconstrucción de imagen")


# Aunque no tengamos las otras páginas aún, Streamlit necesita saber cómo las manejará.
# Por ahora, simplemente las dejamos vacías. Más adelante las llenaremos.
if "page" not in st.session_state:
    st.session_state.page = "Inicio"

# Esto se usará cuando tengamos las otras páginas.
# if st.session_state.page == "Gráficos de Comparación":
#     st.write("Contenido de Gráficos de Comparación")
# elif st.session_state.page == "Reconstrucción de Imágenes":
#     st.write("Contenido de Reconstrucción de Imágenes")