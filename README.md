# Clasificación de Pisos Térmicos de Colombia 🌡️

Actividades de **Machine Learning** de clasificación supervisada, en las que se comparan **5 algoritmos** entrenados y evaluados bajo las mismas condiciones, con especial énfasis en la **prevención de la fuga de información** (*data leakage*).

---

## Índice de notebooks

| Notebook | Contenido |
|---|---|
| [`clasificacion_pisos_termicos_colombia.ipynb`](clasificacion_pisos_termicos_colombia.ipynb) | Clasificación de **pisos térmicos** (Cálido, Templado, Frío) usando datos reales de temperatura de ciudades colombianas (Open-Meteo). |
| [`actividad_clasificacion_ml.ipynb`](actividad_clasificacion_ml.ipynb) | Actividad introductoria de clasificación sobre el dataset **Iris**. |

---

## Algoritmos implementados

1. **Árbol de Decisión** (*Decision Tree*)
2. **Bosque Aleatorio** (*Random Forest*)
3. **Máquina de Vectores de Soporte** (*SVM*)
4. **K-Vecinos más Cercanos** (*K-NN*)
5. **Clasificador Bayesiano Ingenuo** (*Naive Bayes*)

Todos se entrenan y evalúan sobre los mismos datos y con la misma partición, garantizando una comparación justa.

---

## Notebook: Clasificación de pisos térmicos 🌡️

### Objetivo

Predecir el **piso térmico** de un registro diario en Colombia a partir de sus características geográficas y temporales.

- **Fuente de datos**: API pública de [Open-Meteo](https://open-meteo.com/) (sin clave de acceso). Se descargan datos reales de temperatura media diaria del año **2023**.
- **Ciudades**: 26 ciudades colombianas del piso cálido, templado y frío.

### Clases

Las clases se asignan según la **temperatura media diaria** de cada registro:

| Piso térmico | Rango de temperatura | Ejemplos |
|---|---|---|
| ☀️ **Cálido** | > 24 °C | Cartagena, Barranquilla, Cali |
| 🌤️ **Templado** | 18 – 24 °C | Medellín, Bucaramanga, Armenia |
| ❄️ **Frío** | < 18 °C | Bogotá, Pasto, Tunja |

> ⚠️ **Aclaración importante:** las clases se asignan según la **temperatura media diaria** y, por tanto, **no representan necesariamente el piso térmico geográfico oficial** de la ciudad. Son etiquetas *operativas* construidas a partir de la temperatura de ese día específico, lo que permite usar series temporales de temperatura para un ejercicio de clasificación supervisada.

### Características de entrada (`X`)

- `latitud`, `longitud`
- `altitud` (msnm)
- `mes`, `dia_del_anio`
- `precipitacion_mm`

> La **temperatura no se usa como característica** de entrada; se usa únicamente para construir la etiqueta.

### Control de fuga de información 🛡️

1. **División por ciudad** (`GroupShuffleSplit`): una misma ciudad **nunca** aparece a la vez en entrenamiento y prueba. Como cada ciudad aporta ~365 registros diarios con geografía idéntica, un split aleatorio por filas permitiría al modelo "memorizar" una ciudad y evaluarse sobre sus propios días. Separar por ciudad obliga a generalizar a geografía **no vista** (80 % de las ciudades para entrenar / 20 % para probar).
2. **Escalado ajustado solo con entrenamiento**: el `StandardScaler` se ajusta únicamente con las ciudades de entrenamiento y luego se aplica a ambos conjuntos.

### Resultados reales

Resultados obtenidos al ejecutar el notebook (con división por ciudad), ordenados por **F1-Score macro**:

| Modelo | Accuracy | F1 (macro) |
|---|---|---|
| Random Forest | 0.9297 | 0.9290 |
| Naive Bayes | 0.9256 | 0.9263 |
| Árbol de Decisión | 0.8963 | 0.8977 |
| SVM | 0.8813 | 0.8833 |
| K-NN | 0.7945 | 0.7981 |

> ℹ️ Los valores son más bajos que con un split aleatorio por filas; esto es **esperado y deseable**, porque separar por ciudad evalúa la capacidad real de generalización del modelo.

---

## Notebook: Actividad de clasificación (Iris) 🌸

Actividad introductoria que compara los mismos 5 algoritmos sobre el dataset clásico **Iris** (150 muestras, 3 especies), con 80 % entrenamiento / 20 % prueba estratificados y escalado sin fuga de información.

---

## Requisitos

- Python 3.10+
- Paquetes:
  - `numpy`, `pandas`
  - `scikit-learn`
  - `matplotlib`, `seaborn`
  - `requests` (solo para el notebook de pisos térmicos)

Instalación:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn requests
```

> El notebook de pisos térmicos requiere **conexión a Internet** para descargar los datos desde Open-Meteo.

---

## Uso

Abrir los notebooks con Jupyter y ejecutar todas las celdas de arriba hacia abajo:

```bash
jupyter notebook
```

---

## Métricas utilizadas

- **Accuracy** — proporción global de aciertos.
- **Precision** — proporción de predicciones correctas por clase (evita falsos positivos).
- **Recall** — proporción de instancias reales detectadas (evita falsos negativos).
- **F1-Score** — media armónica entre precision y recall.
- **Matriz de confusión** — aciertos y errores por clase.
- **Classification report** — resumen por clase (multiclase, promedios `macro` y `weighted`).
