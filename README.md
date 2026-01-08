# U-net


Una U-Net es una arquitectura de red neuronal convolucional diseñada para segmentación semántica píxel a píxel: dado una imagen de entrada, el modelo predice para cada píxel si pertenece (o no) a una clase, en tu caso una máscara binaria.
​
En este proyecto se usa una U-Net para segmentar imágenes 512x512 a partir de datos RGB y sus máscaras en escala de grises.
​

Estructura de la U-Net
La U-Net tiene forma de “U” porque combina dos partes: encoder (contracción) y decoder (expansión), unidas por conexiones de salto (skip connections).
​

Encoder (camino descendente):

Secuencia de bloques convolucionales con filtros 3x3, normalización por lotes y activación ReLU, seguidos de max pooling 2x2.
​

Cada vez que se aplica pooling:

Se reduce la resolución espacial (altura y anchura se dividen entre 2).

Se duplican los canales de características (más filtros).
​

El encoder aprende representaciones cada vez más abstractas pero con menos resolución espacial.

Bottleneck:

Bloque convolucional con el máximo número de filtros, situado en la parte más profunda de la “U”.
​

Resume la información global de la imagen antes de empezar a reconstruir la máscara.

Decoder (camino ascendente):

Comienza con convoluciones transpuestas (ConvTranspose2d) para hacer “up‑sampling” y recuperar resolución.
​

Cada bloque del decoder:

Sube la resolución espacial (se multiplica por 2).

Reduce el número de canales a la mitad.
​

Concatena (por canales) el mapa de activaciones correspondiente del encoder (skip connection) y después aplica un bloque convolucional.
​

Skip connections y su importancia
Las skip connections conectan directamente la salida de cada bloque del encoder con el bloque correspondiente en el decoder de la misma “escala”.
​

Sin estas conexiones, el decoder solo vería una representación muy comprimida y difusa, perdiendo detalles finos de bordes.

Con los saltos:

Se combinan características de alto nivel (del bottleneck) con detalles locales de baja profundidad (primeros niveles del encoder).

El modelo puede recuperar contornos nítidos y estructuras pequeñas de la máscara.
​

Función de pérdida y entrenamiento
En tu código se usa nn.MSELoss, es decir un error cuadrático medio entre la máscara predicha y la máscara real.
​

Para cada píxel:

El modelo produce un valor continuo (generalmente entre 0 y 1 tras una normalización implícita).

Se compara con el valor de la máscara (0 o 1) y se calcula 
(
y
pred
−
y
true
)
2
(y 
pred
 −y 
true
 ) 
2
 .
​

El entrenamiento:

Usa el optimizador Adam con una tasa de aprendizaje pequeña y weight_decay para regularización.
​

Se ejecuta durante 100 épocas, observándose cómo la pérdida desciende desde valores altos hasta del orden de 
10
−
3
10 
−3
 , señal de ajuste progresivo.
​

En segmentación binaria suele ser común usar también funciones como BCE o Dice Loss, pero en tu notebook se mantiene MSE para simplificar.
​

Flujo completo de datos
El Dataset:

Lee imágenes PNG en RGB y máscaras PNG en escala de grises desde directorios separados.
​

Aplica un Resize(512, 512) a ambos tensores para tener tamaño uniforme.
​

Normaliza dividiendo por 255 y convierte a tensores float32.
​

El DataLoader:

Agrupa ejemplos en lotes (por ejemplo de tamaño 4) para aprovechar el paralelismo del GPU o CPU.
​

Bucle de entrenamiento:

Para cada lote:

Se hace forward: imagen → U-Net → máscara predicha.

Se calcula la pérdida MSE con la máscara real.

Se hace backward y se actualizan parámetros con Adam.

Se imprime “Lote 0, Epoch X/100, Loss Y” para monitorizar el proceso.
​

Este conjunto de ideas te da tanto el contexto práctico del código como la intuición teórica de por qué la U-Net funciona bien para segmentación semántica: combina contracción para entender el “qué” y expansión con saltos para recuperar el “dónde”.
