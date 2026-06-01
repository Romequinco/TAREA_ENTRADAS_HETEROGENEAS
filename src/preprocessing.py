"""
preprocessing.py — Pipeline de carga, limpieza y preparación de datos.

Flujo esperado (ejecutar en este orden):
    1. load_raw_data        → DataFrames crudos
    2. clean_train          → elimina Open==0 y anomalías
    3. merge_store_features → une store.csv, imputa nulos
    4. encode_categoricals  → label-encoding de categóricas
    5. engineer_features    → features temporales cíclicas y de competencia
    6. temporal_split       → tres DataFrames (train / val / test)
    7. build_store_normalizer → fit sobre df_train únicamente
    8. normalize_sales      → aplica log1p + z-score por tienda
    9. build_sequences      → genera arrays listos para el modelo

Todas las fechas de corte se importan de evaluate.py para garantizar
coherencia con la evaluación.
"""

import numpy as np
import pandas as pd

from src.evaluate import (
    SPLIT_TRAIN_END,
    SPLIT_VAL_START,
    SPLIT_VAL_END,
    SPLIT_TEST_START,
)

# Longitud de secuencia por defecto para el LSTM (días abiertos consecutivos)
DEFAULT_SEQ_LEN = 30

# Columnas que entran en la rama recurrente (orden fijo — no alterar)
SEQ_FEATURE_COLS = [
    "Sales_norm",    # target desplazado un paso (teacher forcing en train)
    "Promo",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
    "SchoolHoliday",
    "StateHoliday_enc",
]

# Columnas numéricas que entran en la rama densa
STATIC_NUM_COLS = [
    "CompetitionDistance_norm",
    "competition_age_years",
    "has_competition",
    "Promo2",
]

# Columnas categóricas que entran en la rama densa (como índices enteros)
STATIC_CAT_COLS = [
    "Store_enc",        # embedding dim 16 en model.py
    "StoreType_enc",    # embedding dim 2
    "Assortment_enc",   # embedding dim 2
    "PromoInterval_enc",# embedding dim 2
]


def load_raw_data(data_dir: str) -> tuple:
    """
    Carga train.csv y store.csv desde data_dir.

    Parameters
    ----------
    data_dir : str
        Ruta a la carpeta data/ (p.ej. '../data').

    Returns
    -------
    (train_df, store_df) : tuple de DataFrames
        train_df  — 1.001.599 filas, columna Date como datetime64.
        store_df  — 1.115 filas con metadatos de tiendas.
    """
    raise NotImplementedError


def clean_train(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas que no deben entrar en el modelo:
      - Open == 0  (ventas siempre 0, 17% del dataset)
      - Sales == 0 con Open == 1  (54 casos, ruido/errores de registro)

    Parameters
    ----------
    train_df : pd.DataFrame
        DataFrame crudo de train.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame limpio (~830.972 filas).
    """
    raise NotImplementedError


def merge_store_features(train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une store.csv a train_df por Store e imputa nulos:
      - CompetitionDistance: mediana global (3 nulos).
      - CompetitionOpenSinceYear/Month: 0 cuando es NaN (sin competencia conocida).
      - Promo2SinceWeek/Year y PromoInterval: solo para Promo2==1; NaN en Promo2==0 es correcto.

    Parameters
    ----------
    train_df : pd.DataFrame
        Resultado de clean_train().
    store_df : pd.DataFrame
        DataFrame crudo de store.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas de store añadidas (~830.972 filas, +9 columnas).
    """
    raise NotImplementedError


def encode_categoricals(df: pd.DataFrame) -> tuple:
    """
    Label-encoding de variables categóricas en columnas *_enc.

    Codifica: Store → Store_enc, StoreType → StoreType_enc,
              Assortment → Assortment_enc, PromoInterval → PromoInterval_enc,
              StateHoliday → StateHoliday_enc.

    Los índices empiezan en 0 y son continuos (necesario para Embedding de Keras).

    Parameters
    ----------
    df : pd.DataFrame
        Resultado de merge_store_features().

    Returns
    -------
    (df, encoders) : tuple
        df       — DataFrame con columnas *_enc añadidas.
        encoders — dict {col_name: LabelEncoder} para invertir si hace falta.
    """
    raise NotImplementedError


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features temporales y de competencia.

    Features temporales (codificación cíclica sin/cos):
      - DayOfWeek_sin, DayOfWeek_cos  (período 7)
      - month_sin,     month_cos       (período 12)
      - week_sin,      week_cos        (período 52)

    Features de competencia:
      - has_competition : 1 si CompetitionOpenSinceYear no era nulo original, 0 si no.
      - competition_age_years : años desde apertura competencia hasta la fecha del registro
                                (clip a [0, 20]). 0 si has_competition == 0.

    Features de tendencia:
      - days_since_start : días desde 2013-01-01 (captura tendencia lineal).

    Parameters
    ----------
    df : pd.DataFrame
        Resultado de encode_categoricals().

    Returns
    -------
    pd.DataFrame
        DataFrame con las nuevas columnas añadidas.
    """
    raise NotImplementedError


def temporal_split(df: pd.DataFrame) -> tuple:
    """
    Divide df en tres splits por fecha usando las constantes de evaluate.py.

      - train : Date <= SPLIT_TRAIN_END  (hasta 2014-09-30)
      - val   : SPLIT_VAL_START <= Date <= SPLIT_VAL_END  (oct–dic 2014)
      - test  : Date >= SPLIT_TEST_START  (desde 2015-01-01)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completamente preprocesado.

    Returns
    -------
    (df_train, df_val, df_test) : tuple de DataFrames
        Tamaños aproximados: 648.360 / 75.860 / 182.612 filas.
    """
    raise NotImplementedError


def build_store_normalizer(df_train: pd.DataFrame) -> dict:
    """
    Calcula media y std de log1p(Sales) por tienda, ajustando SOLO sobre df_train.

    La normalización es log1p + z-score por tienda:
        Sales_norm = (log1p(Sales) - mean_store) / std_store

    Importante: se ajusta solo con df_train para evitar data leakage
    en val y test.

    Parameters
    ----------
    df_train : pd.DataFrame
        Split de entrenamiento (resultado de temporal_split()[0]).

    Returns
    -------
    dict
        {store_id: {'mean': float, 'std': float}}
        Para cada tienda presente en df_train.
    """
    raise NotImplementedError


def normalize_sales(df: pd.DataFrame, store_stats: dict) -> pd.DataFrame:
    """
    Aplica la normalización log1p + z-score por tienda usando store_stats.

    Añade la columna 'Sales_norm' al DataFrame.
    Los stats de store_stats proceden siempre de build_store_normalizer(df_train)
    para val y test.

    Parameters
    ----------
    df : pd.DataFrame
        Cualquiera de los tres splits.
    store_stats : dict
        Resultado de build_store_normalizer().

    Returns
    -------
    pd.DataFrame
        DataFrame con columna 'Sales_norm' añadida.
    """
    raise NotImplementedError


def build_sequences(df: pd.DataFrame, seq_len: int = DEFAULT_SEQ_LEN) -> tuple:
    """
    Construye las secuencias de entrada para el modelo many-to-one.

    Para cada tienda, ordena por fecha y genera ventanas deslizantes de
    longitud seq_len sobre los días abiertos (todos los que llegan aquí
    ya son Open==1 gracias a clean_train).

    La etiqueta y[i] es Sales_norm del día i+seq_len (el día a predecir).
    Las features de secuencia X_seq[i] son las filas [i, i+seq_len) de SEQ_FEATURE_COLS.
    Las features estáticas X_static[i] son las columnas STATIC_CAT_COLS + STATIC_NUM_COLS
    del día i+seq_len (son constantes por tienda, pero se indexa así por consistencia).

    SIN lag features explícitos: la secuencia de Sales_norm ya actúa como
    contexto histórico (decisión cerrada en EDA — ver README).

    Parameters
    ----------
    df : pd.DataFrame
        Split ya normalizado (contiene 'Sales_norm').
    seq_len : int
        Longitud de la ventana de contexto (días abiertos). Por defecto 30.

    Returns
    -------
    (X_seq, X_static, y) : tuple de np.ndarray
        X_seq   : shape (n_samples, seq_len, len(SEQ_FEATURE_COLS))  — float32
        X_static: shape (n_samples, len(STATIC_CAT_COLS + STATIC_NUM_COLS))  — float32
                  Las primeras len(STATIC_CAT_COLS) columnas son índices enteros (para Embedding).
                  Las siguientes len(STATIC_NUM_COLS) columnas son numéricas normalizadas.
        y       : shape (n_samples,)  — float32, Sales_norm del día objetivo.
    """
    raise NotImplementedError
