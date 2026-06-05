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

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.evaluate import (
    SPLIT_TRAIN_END,
    SPLIT_VAL_START,
    SPLIT_VAL_END,
    SPLIT_TEST_START,
)

DEFAULT_SEQ_LEN = 30

# Columnas que entran en la rama recurrente (orden fijo — no alterar)
SEQ_FEATURE_COLS = [
    "Sales_norm",
    "Promo",
    "DayOfWeek_sin", "DayOfWeek_cos",
    "month_sin",     "month_cos",
    "week_sin",      "week_cos",
    "SchoolHoliday",
    "StateHoliday_enc",
    "days_since_start",  # tendencia lineal — crítica para tienda 769
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
    "Store_enc",         # embedding dim 16
    "StoreType_enc",     # embedding dim 2
    "Assortment_enc",    # embedding dim 2
    "PromoInterval_enc", # embedding dim 2
]


def load_raw_data(data_dir: str) -> tuple:
    """Carga train.csv y store.csv desde data_dir."""
    train_df = pd.read_csv(
        os.path.join(data_dir, "train.csv"),
        parse_dates=["Date"],
        low_memory=False,
    )
    store_df = pd.read_csv(os.path.join(data_dir, "store.csv"))
    return train_df, store_df


def clean_train(train_df: pd.DataFrame) -> pd.DataFrame:
    """Elimina días cerrados (Open==0) y los 54 registros con Sales==0 estando abierto."""
    return train_df[(train_df["Open"] == 1) & (train_df["Sales"] > 0)].copy()


def merge_store_features(train_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """
    Une store.csv a train_df e imputa nulos.
    CompetitionOpenSinceYear/Month → 0 cuando NaN (sin competencia conocida).
    """
    df = train_df.merge(store_df, on="Store", how="left")
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
    # PromoInterval NaN (tiendas con Promo2==0) se mantiene: se codifica como 'None'
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple:
    """
    Label-encoding de variables categóricas → columnas *_enc (índices 0-based).
    Devuelve (df_con_enc, dict_encoders).
    """
    encoders = {}
    for col in ["Store", "StoreType", "Assortment", "PromoInterval", "StateHoliday"]:
        le = LabelEncoder()
        # Normalizar a str para manejar NaN, mixed-types y el '0' string de StateHoliday
        df[f"{col}_enc"] = le.fit_transform(df[col].fillna("None").astype(str))
        encoders[col] = le
    return df, encoders


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade features temporales cíclicas, de competencia y tendencia lineal."""
    df = df.copy()

    # Features cíclicas: sin/cos evitan la discontinuidad en los extremos del período
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["Date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Date"].dt.month / 12)
    week = df["Date"].dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * week / 52)
    df["week_cos"] = np.cos(2 * np.pi * week / 52)

    # Competencia — year==0 significa "sin competencia conocida" (imputado en merge)
    df["has_competition"] = (df["CompetitionOpenSinceYear"] != 0).astype(np.int8)
    comp_year  = df["CompetitionOpenSinceYear"].replace(0, 1900).astype(int)
    comp_month = df["CompetitionOpenSinceMonth"].clip(lower=1).replace(0, 1).astype(int)
    comp_open  = pd.to_datetime(
        {"year": comp_year, "month": comp_month, "day": 1}, errors="coerce"
    )
    age = (df["Date"] - comp_open).dt.days / 365.25
    df["competition_age_years"] = (
        age.clip(0, 20).fillna(0) * df["has_competition"]
    ).astype(np.float32)

    # Tendencia lineal global (días desde inicio del dataset)
    df["days_since_start"] = (df["Date"] - pd.Timestamp("2013-01-01")).dt.days

    # CompetitionDistance normalizada (feature estática → normalización global no introduce leakage temporal)
    med = df["CompetitionDistance"].median()
    std = df["CompetitionDistance"].std()
    df["CompetitionDistance_norm"] = ((df["CompetitionDistance"] - med) / std).astype(np.float32)

    return df


def temporal_split(df: pd.DataFrame) -> tuple:
    """Divide en train / val / test usando las fechas de evaluate.py."""
    train = df[df["Date"] <= SPLIT_TRAIN_END].copy()
    val   = df[(df["Date"] >= SPLIT_VAL_START) & (df["Date"] <= SPLIT_VAL_END)].copy()
    test  = df[df["Date"] >= SPLIT_TEST_START].copy()
    return train, val, test


def build_store_normalizer(df_train: pd.DataFrame) -> dict:
    """
    Calcula media y std de log1p(Sales) por tienda, ajustando SOLO sobre df_train.
    Devuelve {store_id: {'mean': float, 'std': float}}.
    """
    stats = {}
    for store_id, grp in df_train.groupby("Store"):
        log_s = np.log1p(grp["Sales"])
        stats[int(store_id)] = {"mean": float(log_s.mean()), "std": float(log_s.std())}
    return stats


def normalize_sales(df: pd.DataFrame, store_stats: dict) -> pd.DataFrame:
    """Añade columna 'Sales_norm' = (log1p(Sales) - mean_store) / std_store."""
    df = df.copy()
    means = df["Store"].map({k: v["mean"] for k, v in store_stats.items()})
    stds  = df["Store"].map({k: max(v["std"], 1e-8) for k, v in store_stats.items()})
    df["Sales_norm"] = ((np.log1p(df["Sales"]) - means) / stds).astype(np.float32)
    return df


def build_sequences(df: pd.DataFrame, seq_len: int = DEFAULT_SEQ_LEN) -> tuple:
    """
    Construye ventanas deslizantes para el modelo many-to-one.

    Para cada tienda (ordenada por fecha):
        X_seq[i]    = SEQ_FEATURE_COLS en filas [i, i+seq_len)
        X_static[i] = STATIC_CAT_COLS + STATIC_NUM_COLS del día i+seq_len
        y[i]        = Sales_norm del día i+seq_len

    Devuelve (X_seq, X_static, y) como arrays float32.
    Shapes: (N, seq_len, n_seq_feats), (N, n_static), (N,).
    """
    static_cols = STATIC_CAT_COLS + STATIC_NUM_COLS
    seq_parts, stat_parts, y_parts = [], [], []

    for _, grp in df.groupby("Store"):
        grp = grp.sort_values("Date")
        n = len(grp)
        if n <= seq_len:
            continue

        seq  = grp[SEQ_FEATURE_COLS].to_numpy(dtype=np.float32)
        stat = grp[static_cols].to_numpy(dtype=np.float32)
        y    = grp["Sales_norm"].to_numpy(dtype=np.float32)

        # sliding_window_view → (n-seq_len+1, n_feats, seq_len); [:-1] descarta la última ventana
        # (no tiene label) y transpose convierte a (n-seq_len, seq_len, n_feats)
        windows = np.lib.stride_tricks.sliding_window_view(seq, seq_len, axis=0)
        windows = windows[:-1].transpose(0, 2, 1).copy()

        seq_parts.append(windows)
        stat_parts.append(stat[seq_len:])
        y_parts.append(y[seq_len:])

    return (
        np.concatenate(seq_parts,  axis=0),
        np.concatenate(stat_parts, axis=0),
        np.concatenate(y_parts,    axis=0),
    )
