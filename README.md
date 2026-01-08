# U-net

# U-Net para Segmentación Semántica en PyTorch

Este repositorio contiene la implementación de una **U-Net** en PyTorch para segmentación binaria píxel a píxel a partir de imágenes RGB y sus máscaras en escala de grises. [file:2]

## Descripción general

- Arquitectura tipo U-Net con:
  - Camino de **encoder** (contracción) mediante bloques convolucionales y max pooling.
  - Bottleneck con el mayor número de canales.
  - Camino de **decoder** (expansión) con `ConvTranspose2d` y *skip connections* desde el encoder.
  - Capa final `Conv2d` con 1 canal de salida para la máscara binaria. [file:2]
- Entrenamiento de modelos completos y varias versiones simplificadas (menos filtros) para estudiar la relación entre capacidad del modelo, tiempo de cómputo y calidad de segmentación. [file:2]

## Estructura del código

- **Dataset personalizado**:
  - Lee imágenes desde `datadir` en RGB y máscaras desde `labeldir` en escala de grises, ambas en formato PNG.
  - Aplica `Resize(512, 512)` tanto a la imagen como a la máscara.
  - Normaliza dividiendo por 255 y devuelve tensores `float32`. [file:2]

- **Modelo U-Net**:
  - `convblock`: dos convoluciones 3×3 con batch normalization y ReLU.
  - `encoderblock`: `convblock` + `MaxPool2d`, duplicando canales y reduciendo resolución a la mitad.
  - `decoderblock`: `ConvTranspose2d` para *up-sampling*, concatenación del *skip* del encoder y `convblock`.
  - `buildunet`: ensambla encoder, bottleneck, decoder y la capa de salida `Conv2d(…, 1, 1)`. [file:2]

- **Entrenamiento**:
  - Pérdida: `nn.MSELoss` entre la máscara predicha y la máscara real.
  - Optimizador: `Adam` con `lr=1e-3` y `weight_decay=1e-5`.
  - Entrenamiento típico: 100 épocas, iterando sobre un `DataLoader` con tamaño de batch 4 (ejemplo del cuaderno). [file:2]

- **Visualización**:
  - Función `visualizarpredicmodel` que muestra, para varias muestras:
    - Imagen original.
    - Máscara real (*ground truth*).
    - Predicción del modelo, aplicando un umbral de 0.5 para visualizar la máscara binaria. [file:2]

## Ejecución

1. Preparar el dataset:
   - Carpeta de imágenes RGB: `datadir` (por ejemplo `./test/`).
   - Carpeta de máscaras en escala de grises: `labeldir` (por ejemplo `./labels/`).
   - Mismo nombre de archivo para imagen y máscara (ej. `001.png` en ambos directorios). [file:2]

2. Ajustar rutas en el notebook:
   ```python
   ds = Dataset('./test', './labels')
   dsloader = DataLoader(ds, batch_size=4)



 ## Análisis de resultados
Evolución de la pérdida
- Para el modelo base (U-Net con 64 filtros iniciales), la pérdida MSE en el lote mostrado desciende aproximadamente de 0.20 en la época 1 a valores cercanos a 1.7×10⁻³ en la época 100. [file:2]

- Las versiones simplificadas (con 32, 16 y 8 filtros iniciales) muestran patrones similares, comenzando entre 0.17 y 0.24 y convergiendo también a una pérdida en torno a (1.6–2.0)×10⁻³ tras 100 épocas. [file:2]

En conjunto, las curvas sugieren:

- Convergencia estable: la pérdida decrece de forma mayoritariamente monótona a lo largo del entrenamiento, con pequeñas oscilaciones esperables con Adam. 

- Poca diferencia de MSE final entre el modelo completo y los simplificados, lo que indica que, para el dataset y la configuración usados, incluso arquitecturas con menos filtros aproximan bien la máscara. 

Calidad visual de las predicciones
- Las figuras de visualizarpredicmodel muestran que:

    - La máscara predicha reproduce correctamente la forma global de la región de interés.

    - Los contornos se alinean razonablemente con la máscara real, con pequeñas discrepancias en detalles finos. [file:2]

- Las versiones simplificadas mantienen una segmentación cualitativamente buena; el modelo más grande tiende a producir bordes algo más definidos, pero las diferencias no son drásticas en los ejemplos mostrados. [file:2]

Interpretación
- La combinación de MSE baja y buena coincidencia visual sugiere que el modelo está bien ajustado al conjunto de datos, capturando tanto la localización como la forma de los objetos segmentados. [file:2]

- La similitud en las pérdidas finales entre las distintas simplificaciones indica que:

    - El problema probablemente no es extremadamente complejo o el dataset es relativamente sencillo.

    - Es posible reducir el número de filtros (y por tanto parámetros y coste computacional) sin perder demasiado rendimiento, útil para despliegue en hardware limitado. 
