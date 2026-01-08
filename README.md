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
