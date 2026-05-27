# Proy_IA_I

![](banner.PNG)

## Objetivo
El objetivo de este proyecto es construir un conjunto de datos (*dataset*) de características estadísticas a partir de datos de alta frecuencia del mercado de divisas, con el fin de entrenar modelos de aprendizaje automático capaces de clasificar el régimen cambiario asociado a cada moneda.

El régimen cambiario es una propiedad del país emisor de la divisa e incluye las siguientes categorías:
* **Free Float** (Flotación libre)
* **Managed Float** (Flotación administrada)
* **High Intervention** (Alta intervención institucional)
* **Fixed Peg** (Tipo de cambio fijo)

El *dataset* final describe el comportamiento estadístico de cada moneda en ventanas temporales de mercado.

---

## 1. Construcción de la Base de Datos y Preprocesamiento
La base de datos se construyó utilizando la API de **Exness**, calculando la fuerza relativa de las 8 monedas más relevantes del mercado. Los datos secuenciales se analizaron y agruparon para eliminar la dependencia temporal, permitiendo construir variables que describieran los comportamientos no lineales inherentes a la naturaleza de los datos.


#  Formación dataset

1. Extracción de pares de divisas desde API de MT5 con python. Se tomaron 28 pares de divisas de las 8 monedas relevantes (USD,EUR,GBP,CAD,NZD,CHF,JPY,AUD). Cada par de divisa contenia 99999x8  (cierre,apertura,max,min,volumen,volumen_real,spread,fecha)

2. Se reconstruyeron las monedas sinteticas segun la interaccion del mercado global (28 monedas) para tener los datos de cada divisa por separado. cada dataset contiene 99999x7 (valor_relativo,cierre,apertura,max,min,volumen )

USD_rec.csv
EUR_rec.csv
GBP_rec.csv
JPY_rec.csv
AUD_rec.csv
CAD_rec.csv
CHF_rec.csv
NZD_rec.csv

3. Se unificaron en un unico dataset donde se agruparon las estadisticas en conjuntos por pais. cada elemento cuenta con estadisticas correspondientes a cada dia.

4. Los datos secuenciales se analizaron y agruparon para eliminar dependencia temporal y construir variables que permitieran describir comportamientos no lineales comunes de la naturaleza de los datos. Como: 

- Tendencia
- Volatilidad
- Persistencia temporal 
- Actividad de mercado
- Estructura del movimiento


download_url = f"https://drive.google.com/uc?export=download&id 1KM9i4yegggk-WaHAeK1LXxzRmqyXxZGJ"


### Características del Dataset:
* **Volumen:** ~8,500 registros.
* **Dimensiones:** 11 variables predictoras.
* **Estrategia de validación:** Partición de datos 80/20 (Entrenamiento/Prueba).

### Distribución de Clases (Desbalanceo):
El *dataset* presenta un marcado desbalanceo de clases, distribuido de la siguiente manera:

| Clase | Régimen Cambiario | Presencia (%) |
| :---: | :--- | :---: |
| **Clase 0** | Alta Intervención | 8.67% |
| **Clase 1** | Free Float | 66.13% |
| **Clase 2** | Managed Float | 23.26% |
| **Clase 3** | Fixed Peg | 1.93% |

---
### Variables Utilizadas

Aquí se detallan las 11 variables calculadas y utilizadas para capturar los comportamientos estadísticos y no lineales de las divisas:

| Variable | Descripción |
| :--- | :--- |
| **vol_mean_30** | Media de la volatilidad en una ventana de 30 períodos. |
| **vol_std_30** | Desviación estándar de la volatilidad en 30 períodos. |
| **trend_mean_30** | Media de la tendencia en una ventana de 30 períodos. |
| **autocorr_mean_30** | Media de la autocorrelación en 30 períodos. |
| **autocorr_std_30** | Desviación estándar de la autocorrelación en 30 períodos. |
| **trend_persistence_30** | Persistencia de la tendencia en 30 períodos. |
| **autocorr_decay** | Tasa de decaimiento de la autocorrelación. |
| **max_drawdown** | Máxima reducción de precios (*Maximum Drawdown*). |
| **autocorr_sign** | Signo de la autocorrelación. |
| **autocorr_sign_change** | Cambios de signo en la autocorrelación. |
| **autocorr_instability** | Inestabilidad de la autocorrelación. |

## 2. Modelos Supervisados Implementados

### Árbol de Decisión (DT)
* **Configuración:** Profundidad máxima (`max_depth`) de 5.
* **Validación:** Validación cruzada (*Cross-Validation*) óptima con 4 *folds*.
* **Métrica:** *Accuracy* de **0.725**.

### Random Forest (RF)
* **Configuración:** 130 estimadores.
* **Validación:** Validación cruzada óptima con 4 *folds*.
* **Métrica:** *Accuracy* de **0.801** *(el mejor rendimiento entre los modelos clásicos)*.

### Máquinas de Soporte Vectorial (SVM)
* **Configuración:** Se implementaron tres tipos de kernels con modificaciones específicas para acelerar la convergencia.
* **Rendimiento:** Los tres kernels (**Lineal, Polinomial y RBF**) obtuvieron el mismo rendimiento.
* **Métrica:** *Accuracy* de **0.769**.

### Redes Neuronales Profundas (DNN)
Se evaluaron dos arquitecturas de *Deep Learning*:
* **Modelo 1 (Simple):** 4 capas densas (128, 128, 64, 4). **Accuracy: 64.00%**
* **Modelo 2 (Complejo):** 7 capas densas (64, 64, 128, 128, 256, 256, 4). **Accuracy: 84.00%**

> **Nota sobre el Modelo 2:** Este modelo mostró curvas de aprendizaje estables a lo largo de las épocas, sufriendo menos de sobreentrenamiento (*overfitting*). Sin embargo, el análisis de aciertos por clase refleja directamente el impacto del desbalanceo original de los datos:

| Clase | Régimen Cambiario | Aciertos (%) |
| :---: | :--- | :---: |
| **Clase 0** | Alta Intervención | 6.38% |
| **Clase 1** | Free Float | 88.40% |
| **Clase 2** | Managed Float | 98.81% |
| **Clase 3** | Fixed Peg | 6.29% |

*(Se cuenta con el registro de las matrices de confusión y el desempeño del accuracy a lo largo de las épocas para su posterior visualización).*

---

## 3. Análisis No Supervisado
Con el objetivo de explorar la estructura intrínseca de los datos sin etiquetas, se aplicaron técnicas de reducción de dimensionalidad y clustering:

* **Reducción de Dimensionalidad (PCA):** Se utilizó el método PCA para aplanar las dimensiones del *dataset* a 2. Tras la reducción, la estructura visual de los datos parece poco diferenciable y sin presencia de patrones evidentes, lo que dificulta el uso de métodos no supervisados.
* **K-Means:** Al intentar agrupar los datos en 4 sectores, el método del codo (*Elbow Method*) sugirió de manera consistente que el número óptimo de clústeres ($K$) es de **3**.
* **DBSCAN:** Debido a la dispersión y falta de separación clara, el algoritmo no logró detectar patrones de densidad ni fronteras lógicas, clasificando casi la totalidad de la base de datos dentro de una única gran clase.

