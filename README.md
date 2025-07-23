Código utilizado para el artículo «Una red de Hopfield cuantificada dispersa para memoria continua en línea».

Ejecutado con Python 3.7.6 y PyTorch 1.10.0.

Todos los conjuntos de datos, excepto el de TinyImagenet, se descargan automáticamente a través de PyTorch. Para descargar TinyImagenet, consulte https://github.com/tjmoon0104/pytorch-tiny-imagenet?tab=readme-ov-file

Para reproducir datos de una ejecución de entrenamiento/experimento:

main.py --test argument

Para reproducir gráficos de una ejecución de entrenamiento/experimento:

main.py --plot argument

Aquí se muestran los argumentos de los distintos experimentos utilizados para reproducir pruebas y gráficos:

Comparaciones de memoria asociativa: assoc_comp

Online-Continual prueba una capa oculta: OnCont-L1

Online-Continual prueba tres capas ocultas: OnCont-L3

Codificación con ruido prueba una capa oculta: nsEncode-L1

Codificación con ruido prueba una capa oculta: nsEncode-L3

Codificación con ruido prueba una capa oculta: recog

Comparaciones de arquitectura: arch compare

Por ejemplo, para reproducir los gráficos de la tarea Online-Continual con la capa oculta Modelos de capas, ejecutar

main.py --test OnCont-L1

seguido de

main.py --plot OnCont-L1




Streamlit (Para visualizar resultados y probar el restaurador de QR):

```bash
https://trabajofinalml-lztvgkzxqmlx5kzxy6kwnx.streamlit.app
```

Codigo realizado en Colab:

```bash
https://drive.google.com/file/d/1UFkDAIVonXX4UhlfvHL6ZRHkJTFOrIDL/view?usp=sharing
```
