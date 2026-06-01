# Rossmann Sales — Red Neuronal con Entradas Heterogéneas

**Equipo:** Oscar · Dani · Fernando  
**Máster MIAX — Práctica entradas heterogéneas**

---

## Descripción

Red neuronal many-to-one para predecir ventas diarias en 9 tiendas Rossmann.
Arquitectura con dos ramas:
- **Rama recurrente (LSTM):** serie temporal de los últimos 30 días abiertos.
- **Rama densa (Embeddings):** variables estáticas de tienda (tipo, surtido, competencia…).

---

## Estructura del repo

```
.
├── data/                    # CSVs originales — NO se suben a git (.gitignore)
│   ├── train.csv
│   ├── store.csv
│   ├── test.csv
│   └── submission.csv
├── src/
│   ├── evaluate.py          # Contrato de evaluación (fechas, R², TARGET_STORES)
│   ├── preprocessing.py     # Pipeline completo de preprocesado
│   └── model.py             # Arquitectura Keras y entrenamiento
├── notebooks/
│   ├── oscar.ipynb
│   ├── dani.ipynb
│   └── fernando.ipynb
├── outputs/                 # Modelos y predicciones — NO se suben a git
├── práctica.ipynb           # Notebook original del profesor (referencia)
└── requirements.txt
```

---

## Cómo empezar

### 1. Clonar el repo

```bash
git clone <url-del-repo>
cd TAREA_ENTRADAS_HETEROGENEAS
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> En entornos con GPU, instalar `tensorflow[and-cuda]==2.15.0` en lugar de `tensorflow==2.15.0`.

### 3. Obtener los datos

Los CSVs no están en git por su tamaño. Descargarlos del Google Drive compartido
y colocarlos en `data/`:

```
data/train.csv
data/store.csv
data/test.csv
data/submission.csv
```

### 4. Abrir el notebook personal

```bash
jupyter notebook notebooks/oscar.ipynb   # o dani / fernando
```

---

## Flujo de trabajo en git (nivel básico, sin ramas)

**Antes de ponerse a trabajar:**
```bash
git pull
```

**Al terminar una sesión:**
```bash
git add notebooks/oscar.ipynb src/preprocessing.py  # solo los archivos que tocaste
git commit -m "preprocessing: limpieza y encode categoricals"
git push
```

Reglas acordadas:
- `git pull` siempre antes de empezar. Sin excepción.
- Commits frecuentes y pequeños (mejor 5 commits pequeños que 1 grande al final).
- Nunca hacer `git add .` — añadir solo los archivos que cada uno modificó.
- Si hay conflicto en un notebook: resolverlo manualmente o avisar por WhatsApp.
- `src/evaluate.py` no se toca sin consenso del equipo (es el contrato común).

---

## Decisiones técnicas cerradas (EDA — 2026-06-01)

| Decisión | Elección | Razón |
|---|---|---|
| Framework | `tensorflow==2.15.0` (tf.keras) | Portabilidad CPU/GPU, notebook original ya en Keras |
| Target | `log1p(Sales)` z-score por tienda | Skewness 1.60, escalas muy distintas entre tiendas |
| Filtro Open==0 | Eliminar del training | 17% de datos, Sales=0 por construcción, destruye gradiente |
| Lags explícitos | NO en v1 | La secuencia LSTM ya provee contexto histórico; añadir lags multiplica riesgo de leakage |
| Modelo | Global con Store Embedding | Tiendas 1-5 solo tienen ~600 días de train, insuficiente para modelo individual |
| Validación | 2014-10-01 → 2014-12-31 | Incluye pico navideño, más exigente que periodo "plano" |
| Test | 2015-01-01 → 2015-07-17 | Definido por el enunciado (split interno sobre train.csv) |
| SEQ_LEN | 30 días abiertos | Captura ~6 semanas de historia, manejable en CPU |

### Split temporal

```
2013-01-01 ──────── 2014-09-30 | 2014-10-01 ── 2014-12-31 | 2015-01-01 ── 2015-07-17
        TRAIN (648 K filas)          VAL (76 K)                TEST (183 K) ← evaluado
```

### Tiendas objetivo

```python
TARGET_STORES = [1, 2, 3, 4, 5, 562, 682, 733, 769]
```

Notas del EDA relevantes para el modelo:
- Tiendas 1-5 cierran los domingos; 562/682/733/769 abren 7 días.
- Tienda 769: tendencia creciente en 2015 (ratio test/train = 1.17) — incluir `days_since_start`.
- Promo aumenta ventas +38% → feature más importante tras el ID de tienda.
- Diciembre es pico universal (+24–36%) → validación oct-dic lo captura.

---

## Reparto de trabajo

| Persona | Fichero principal | Funciones a implementar |
|---|---|---|
| **Oscar** | `src/preprocessing.py` | `load_raw_data`, `clean_train`, `merge_store_features`, `encode_categoricals`, `engineer_features`, `temporal_split`, `build_store_normalizer`, `normalize_sales`, `build_sequences` |
| **Dani** | `src/model.py` | `build_model` (arquitectura LSTM + Embeddings), `get_callbacks`, `train_model`, `load_trained_model` |
| **Fernando** | `src/evaluate.py` + análisis | `r2_per_store`, `r2_global`, `evaluation_report` + visualizaciones en su notebook |

El notebook de integración (`INTEGRACION.ipynb`, a crear cuando los stubs estén implementados)
llama a las funciones de los tres módulos en secuencia.
