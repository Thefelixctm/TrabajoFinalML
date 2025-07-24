#  SQHN: Sparse Quantized Hopfield Network con Autoencoder y Lector de QR

Este repositorio contiene una versión extendida del proyecto **“A Sparse Quantized Hopfield Network for Online-Continual Memory”**, con nuevas funcionalidades que incluyen un **Autoencoder** para compresión de datos y un **lector/restaurador de códigos QR**. Se incorpora una interfaz visual desarrollada en Streamlit y un entorno práctico en Google Colab.

---

##  Características

-  Implementación base de la Red de Hopfield Cuantificada Dispersa (SQHN)
-  Evaluación de tareas de memoria continua y codificación con ruido
-  Añadido módulo de **Autoencoder**
-  Módulo de **lectura y restauración de códigos QR**
-  Visualización interactiva con **Streamlit**
-  Compatible con ejecución en **Google Colab**

---

##  Requisitos

- Python 3.7.6  
- PyTorch 1.10.0  
- OpenCV, NumPy, Matplotlib, Streamlit


##  Datasets

- Todos los datasets (excepto TinyImageNet) se descargan automáticamente con PyTorch.
- Para usar **TinyImageNet**, debes descargarlo manualmente desde:  
   https://github.com/tjmoon0104/pytorch-tiny-imagenet

---

##  Ejecución de Experimentos

Para entrenar un modelo o reproducir experimentos desde terminal:

```bash
python main.py --test <codigo_experimento>
```

Para visualizar gráficos:

```bash
python main.py --plot <codigo_experimento>
```

### Argumentos disponibles

| Descripción del Experimento             | Argumento      |
|----------------------------------------|----------------|
| Comparación de memoria asociativa      | assoc_comp     |
| Memoria continua (1 capa oculta)       | OnCont-L1      |
| Memoria continua (3 capas ocultas)     | OnCont-L3      |
| Codificación ruidosa (1 capa)          | nsEncode-L1    |
| Codificación ruidosa (3 capas)         | nsEncode-L3    |
| Reconocimiento con ruido               | recog          |
| Comparación de arquitecturas           | arch compare   |

Ejemplo completo:

```bash
python main.py --test OnCont-L1
python main.py --plot OnCont-L1
```

---

##  Streamlit App (Visualización y QR)

Prueba el sistema de recuperación de memoria y el lector de QR en línea:

 [Abrir aplicación Streamlit](https://sqhn-autoencoder.streamlit.app)

NOTA: Para hacer una prueba rapida del lector de QR se debe ocupar la carpeta "QR(Testeo)".

---

##  Notebook en Google Colab

Puedes ejecutar el proyecto completo directamente desde Google Colab:  

 [Abrir en Colab](https://drive.google.com/file/d/1UFkDAIVonXX4UhlfvHL6ZRHkJTFOrIDL/view?usp=sharing)

---

##  Créditos

Basado en el paper original "A Sparse Quantized Hopfield Network for Online-Continual Memory".  
Modificado y extendido con nuevas herramientas de compresión, recuperación visual y análisis interactivo.

Autores: Nicholas Alonso y Jeffrey L. Krichmar

Paper original: https://www.nature.com/articles/s41467-024-46976-4

Github: https://github.com/nalonso2/SQHN
