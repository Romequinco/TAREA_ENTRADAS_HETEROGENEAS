"""
model.py — Arquitectura many-to-one con entradas heterogéneas.

Esquema de la red:
                     ┌──────────────────────────────┐
  X_seq              │  Rama recurrente               │
  (batch,seq,feats)─►│  LSTM → Dropout → Dense        │──┐
                     └──────────────────────────────┘  │
                                                        ├─► Concat → Dense → Dense → Output
                     ┌──────────────────────────────┐  │
  X_static           │  Rama densa                    │  │
  (batch, static) ──►│  Embeddings + Dense            │──┘
                     └──────────────────────────────┘

X_static layout (definido en preprocessing.py):
    cols 0..len(STATIC_CAT_COLS)-1  → índices enteros para Embedding
    cols len(STATIC_CAT_COLS)..     → features numéricas normalizadas
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.preprocessing import STATIC_CAT_COLS, STATIC_NUM_COLS

# Dimensiones de embedding por categórica (mismo orden que STATIC_CAT_COLS)
EMBEDDING_DIMS = {
    "Store_enc":         16,
    "StoreType_enc":      2,
    "Assortment_enc":     2,
    "PromoInterval_enc":  2,
}

# Vocabularios conocidos del dataset Rossmann; Store_enc se pasa como parámetro n_stores
_VOCAB_SIZES = {
    "StoreType_enc":      4,   # a, b, c, d
    "Assortment_enc":     3,   # a, b, c
    "PromoInterval_enc":  4,   # None + 3 intervalos
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
    """Construye y compila el modelo LSTM + Embeddings. Ver docstring del módulo."""
    n_cat  = len(STATIC_CAT_COLS)
    dims   = {**EMBEDDING_DIMS, "Store_enc": store_embedding_dim}
    vocabs = {**_VOCAB_SIZES,   "Store_enc": n_stores}

    inp_seq    = keras.Input((seq_len, n_seq_features), name="seq")
    inp_static = keras.Input((n_cat + n_static_num_features,), name="static")

    # Rama recurrente
    x = layers.LSTM(lstm_units, name="lstm")(inp_seq)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_branch_units, activation="relu")(x)

    # Rama densa — un Embedding por categórica + features numéricas directas
    emb_outs = []
    for i, col in enumerate(STATIC_CAT_COLS):
        # Lambda con i=i captura el índice de bucle correctamente (closure explícita)
        col_slice = layers.Lambda(lambda t, i=i: t[:, i : i + 1])(inp_static)
        emb = layers.Embedding(vocabs[col], dims[col], name=f"emb_{col}")(col_slice)
        emb_outs.append(layers.Flatten()(emb))

    num_feats = layers.Lambda(lambda t: t[:, n_cat:])(inp_static)
    s = layers.Concatenate()(emb_outs + [num_feats])
    s = layers.Dense(dense_branch_units, activation="relu")(s)
    s = layers.Dropout(dropout_rate)(s)

    # Fusión
    merged = layers.Concatenate()([x, s])
    merged = layers.Dense(merge_units, activation="relu")(merged)
    merged = layers.Dropout(dropout_rate)(merged)
    out    = layers.Dense(1, activation="linear", name="output")(merged)

    model = keras.Model(inputs=[inp_seq, inp_static], outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model


def get_callbacks(checkpoint_path: str) -> list:
    """EarlyStopping (p=10) + ReduceLROnPlateau (p=5, ×0.5) + ModelCheckpoint."""
    return [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]


def train_model(
    model: keras.Model,
    X_train: tuple,
    y_train,
    X_val: tuple,
    y_val,
    callbacks: list,
    epochs: int = 50,
    batch_size: int = 256,
) -> keras.callbacks.History:
    """Wrapper de model.fit con las convenciones del proyecto."""
    return model.fit(
        list(X_train),
        y_train,
        validation_data=(list(X_val), y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )


def load_trained_model(path: str) -> keras.Model:
    """Carga un modelo guardado por ModelCheckpoint (formato .h5 o .keras)."""
    return keras.models.load_model(path)
