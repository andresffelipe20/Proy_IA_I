===============================================================================
REPORTE TÉCNICO: CLASIFICACIÓN DE REGÍMENES CAMBIARIOS MEDIANTE APRENDIZAJE AUTOMÁTICO
===============================================================================

OBJETIVO
-------------------------------------------------------------------------------
El objetivo de este proyecto es construir un conjunto de datos (dataset) de
características estadísticas a partir de datos de alta frecuencia del mercado
de divisas. Esto con el fin de entrenar modelos de aprendizaje automático 
capaces de clasificar el régimen cambiario asociado a cada moneda.

El régimen cambiario es una propiedad del país emisor de la divisa e incluye
las siguientes categorías:
* Free Float (Flotación libre)
* Managed Float (Flotación administrada)
* High Intervention (Alta intervención institucional)
* Fixed Peg (Tipo de cambio fijo)

El dataset final describe el comportamiento estadístico de cada moneda en
ventanas temporales de mercado.


1. CONSTRUCCIÓN DE LA BASE DE DATOS Y PREPROCESAMIENTO
-------------------------------------------------------------------------------
La base de datos se construyó utilizando la API de Exness, calculando la fuerza
relativa de las 8 monedas más relevantes del mercado. Los datos secuenciales se
analizaron y agruparon para eliminar la dependencia temporal, permitiendo
construir variables que describieran los comportamientos no lineales inherentes
a la naturaleza de los datos.

Características del Dataset:
* Volumen: ~8,500 registros.
* Dimensiones: 11 variables predictoras.
* Estrategia de validación: Partición de datos 80/20 (Entrenamiento/Prueba).

Distribución de Clases (Desbalanceo):
El dataset presents un marcado desbalanceo de clases, distribuido de la
siguiente manera:

- Clase 0 (Alta Intervención): 8.67% de presencia
- Clase 1 (Free Float):        66.13% de presencia
- Clase 2 (Managed Float):     23.26% de presencia
- Clase 3 (Fixed Peg):         1.93% de presencia


2. MODELOS SUPERVISADOS IMPLEMENTADOS
-------------------------------------------------------------------------------

* Árbol de Decisión (DT)
  - Configuración: Profundidad máxima (max_depth) de 5.
  - Validación: Validación cruzada (Cross-Validation) óptima con 4 folds.
  - Métrica: Accuracy de 0.725.

* Random Forest (RF)
  - Configuración: 130 estimadores.
  - Validación: Validación cruzada óptima con 4 folds.
  - Métrica: Accuracy de 0.801 (el mejor rendimiento entre modelos clásicos).

* Máquinas de Soporte Vectorial (SVM)
  - Configuración: Se implementaron tres tipos de kernels con modificaciones
    específicas para acelerar la convergencia.
  - Rendimiento: Los tres kernels (Lineal, Polinomial y RBF) obtuvieron el
    mismo rendimiento.
  - Métrica: Accuracy de 0.769.

* Redes Neuronales Profundas (DNN)
  Se evaluaron dos arquitecturas de Deep Learning:
  
  - Modelo 1 (Simple): 4 capas densas (128, 128, 64, 4). 
    Accuracy: 64.00%
    
  - Modelo 2 (Complejo): 7 capas densas (64, 64, 128, 128, 256, 256, 4).
    Accuracy: 84.00%

  Nota sobre el Modelo 2:
  Este modelo mostró curvas de aprendizaje estables a lo largo de las épocas,
  sufriendo menos de sobreentrenamiento (overfitting). Sin embargo, el análisis
  de aciertos por clase refleja directamente el impacto del desbalanceo original
  de los datos:
  
  - Clase 0 (Alta intervención):  6.38% de aciertos
  - Clase 1 (Free Float):        88.40% de aciertos
  - Clase 2 (Managed Float):     98.81% de aciertos
  - Clase 3 (Fixed Peg):          6.29% de aciertos

  (Se cuenta con el registro de las matrices de confusión y el desempeño del 
  accuracy a lo largo de las épocas para su posterior visualización).


3. ANÁLISIS NO SUPERVISADO
-------------------------------------------------------------------------------
Con el objetivo de explorar la estructura intrínseca de los datos sin etiquetas,
se aplicaron técnicas de reducción de dimensionalidad y clustering:

* Reducción de Dimensionalidad (PCA):
  Se utilizó el método PCA para aplanar las dimensiones del dataset a 2. Tras
  la reducción, la estructura visual de los datos parece poco diferenciable y 
  sin presencia de patrones evidentes, lo que dificulta el uso de métodos
  no supervisados.

* K-Means:
  Al intentar agrupar los datos en 4 sectores, el método del codo (Elbow Method)
  sugirió de manera consistente que el número óptimo de clústeres (K) es de 3.

* DBSCAN:
  Due a la dispersión y falta de separación clara, el algoritmo no logró
  detectar patrones de densidad ni fronteras lógicas, clasificando casi la
  totalidad de la base de datos dentro de una única gran clase.
===============================================================================