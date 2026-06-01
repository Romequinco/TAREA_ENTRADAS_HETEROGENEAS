"""
model.py — Arquitectura many-to-one con entradas heterogéneas.

Esquema de la red:
                         ┌─────────────────────────────────┐
  X_seq                  │  Rama recurrente                 │
  (batch, seq, feats) ──►│  LSTM → Dropout → Dense          │──┐
                         └─────────────────────────────────┘  │
                                                               ├─► Concatenate
                         ┌─────────────────────────────────┐  │   → Dense → Dense
  X_static               │  Rama densa                      │  │   → Output (1)
  (batch, static) ───────│  Embeddings + Dense layers        │──┘
                         └─────────────────────────────────┘

El modelo devuelve un único valor por muestra (Sales_norm predicho).
La desnormalización a Sales en euros se hace en evaluate.py.

Convención de X_static (definida en preprocessing.py):
    Columnas 0..len(STATIC_CAT_COLS)-1  → índices enteros para Embedding
    Columnas len(STATIC_CAT_COLS)..     → features numéricas normalizadas
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Vocabularios de las variables categóricas estáticas.
# Importar desde preprocessing para mantener una sola fuente.
from src.preprocessing import STATIC_CAT_COLS, STATIC_NUM_COLS

# Dimensiones de embedding para cada categórica estática (en el mismo orden que STATIC_CAT_COLS)
EMBEDDING_DIMS = {
    "Store_enc":         16,   # 1115 tiendas → dim 16
    "StoreType_enc":      2,   # 4 tipos
    "Assortment_enc":     2,   # 3 niveles
    "PromoInterval_enc":  2,   # 3 intervalos + 'sin promo' → 4 clases
}


def build_model(
    n_stores: int,
    seq_len: int,
    n_seq_features: int,
    n_static_num_features: int,
    store_embedding_dim: int = 16,
    lstm_units: int = 64,
    dense_branch_units: int = 64,
    merge_units: int = 128,
    dropout_rate: float = 0.2,
) -> keras.Model:
    """
    Construye el modelo many-to-one con rama LSTM + rama densa de embeddings.

    Rama recurrente
    ---------------
    Input: (batch, seq_len, n_seq_features)
    → LSTM(lstm_units, return_sequences=False)
    → Dropout(dropout_rate)
    → Dense(dense_branch_units, activation='relu')

    Rama densa (estáticos)
    ----------------------
    Input: (batch, n_static_cat + n_static_num)
    Para cada categórica en STATIC_CAT_COLS:
        → Embedding(vocab_size, dim) → Flatten
    Numéricas: pasan directas
    → Concatenate todas
    → Dense(dense_branch_units, activation='relu')
    → Dropout(dropout_rate)

    Fusión
    ------
    Concatenate([salida_LSTM, salida_densa])
    → Dense(merge_units, activation='relu')
    → Dropout(dropout_rate)
    → Dense(1, activation='linear')   ← Sales_norm predicho

    Parameters
    ----------
    n_stores : int
        Cardinalidad de Store (para el Embedding). Normalmente 1115.
    seq_len : int
        Longitud de la secuencia temporal (días abiertos). Debe coincidir
        con el seq_len usado en preprocessing.build_sequences.
    n_seq_features : int
        Número de features por paso de tiempo = len(SEQ_FEATURE_COLS).
    n_static_num_features : int
        Número de features numéricas en X_static = len(STATIC_NUM_COLS).
        Las categóricas se derivan de STATIC_CAT_COLS automáticamente.
    store_embedding_dim : int
        Dimensión del embedding de Store. Sobreescribe EMBEDDING_DIMS['Store_enc'].
    lstm_units : int
        Unidades del LSTM.
    dense_branch_units : int
        Unidades de la capa Dense en cada rama antes de la fusión.
    merge_units : int
        Unidades de la Dense tras la concatenación.
    dropout_rate : float
        Tasa de dropout aplicada en ambas ramas y tras la fusión.

    Returns
    -------
    keras.Model
        Modelo compilado con optimizer Adam y loss MSE.
        Inputs: [input_seq, input_static]
        Output: tensor (batch, 1)
    """
    raise NotImplementedError


def get_callbacks(checkpoint_path: str) -> list:
    """
    Devuelve la lista de callbacks estándar para el entrenamiento.

    Incluye:
      - ModelCheckpoint: guarda el mejor modelo (val_loss) en checkpoint_path.
      - EarlyStopping: paciencia 10 épocas sobre val_loss.
      - ReduceLROnPlateau: reduce LR x0.5 si val_loss no mejora en 5 épocas.

    Parameters
    ----------
    checkpoint_path : str
        Ruta donde guardar los pesos (p.ej. '../outputs/best_model.weights.h5').

    Returns
    -------
    list
        Lista de keras.callbacks.
    """
    raise NotImplementedError


def train_model(
    model: keras.Model,
    X_train: tuple,
    y_train: np.ndarray,
    X_val: tuple,
    y_val: np.ndarray,
    callbacks: list,
    epochs: int = 50,
    batch_size: int = 256,
) -> keras.callbacks.History:
    """
    Entrena el modelo y devuelve el historial.

    Parameters
    ----------
    model : keras.Model
        Modelo construido por build_model().
    X_train : tuple
        (X_seq_train, X_static_train) — arrays numpy de entrenamiento.
    y_train : np.ndarray, shape (n,)
        Target normalizado de entrenamiento.
    X_val : tuple
        (X_seq_val, X_static_val) — arrays numpy de validación.
    y_val : np.ndarray, shape (n,)
        Target normalizado de validación.
    callbacks : list
        Lista de callbacks (resultado de get_callbacks()).
    epochs : int
        Número máximo de épocas (EarlyStopping puede cortar antes).
    batch_size : int
        Tamaño de batch. 256 funciona bien CPU/GPU para este dataset.

    Returns
    -------
    keras.callbacks.History
        Historial de entrenamiento (loss, val_loss por época).
    """
    raise NotImplementedError


def load_trained_model(path: str) -> keras.Model:
    """
    Carga un modelo guardado desde path.

    Parameters
    ----------
    path : str
        Ruta al fichero .weights.h5 o .keras guardado por ModelCheckpoint.

    Returns
    -------
    keras.Model
    """
    raise NotImplementedError
