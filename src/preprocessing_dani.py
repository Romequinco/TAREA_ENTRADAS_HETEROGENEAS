"""
preprocessing_dani.py — Extensiones de preprocesado específicas de Dani.

NO modifica src/preprocessing.py (contrato del equipo). Las funciones de este
módulo se aplican ENCIMA del pipeline base, justo después de normalize_sales
y antes de build_sequences.

Features añadidas a la rama secuencial:
    - Sales_lag_7   : Sales_norm con shift(7) por tienda
    - Sales_lag_14  : Sales_norm con shift(14) por tienda
    - Sales_roll_28 : media móvil 28 días de Sales_norm (shift 1 previo, sin leakage)
"""

import numpy as np
import pandas as pd

from src.preprocessing import (
    STATIC_CAT_COLS,
    STATIC_NUM_COLS,
    DEFAULT_SEQ_LEN,
)

# Mismas features secuenciales del pipeline base, AMPLIADAS con lags y rolling
SEQ_FEATURE_COLS_EXT = [
    "Sales_norm",
    "Promo",
    "DayOfWeek_sin", "DayOfWeek_cos",
    "month_sin",     "month_cos",
    "week_sin",      "week_cos",
    "SchoolHoliday",
    "StateHoliday_enc",
    "days_since_start",
    # --- Nuevas ---
    "Sales_lag_7",
    "Sales_lag_14",
    "Sales_roll_28",
]


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade lag_7, lag_14 y rolling_28 sobre Sales_norm, por tienda.

    IMPORTANTE: usa shift(1) antes del rolling para garantizar que el día t
    solo ve datos hasta t-1 (evita leakage del target sobre sí mismo).

    Los NaN iniciales (primeros días de cada tienda, sin historia suficiente)
    se imputan con 0 (la media de Sales_norm por construcción).
    """
    df = df.sort_values(["Store", "Date"]).copy()

    grp = df.groupby("Store")["Sales_norm"]

    df["Sales_lag_7"]   = grp.shift(7)
    df["Sales_lag_14"]  = grp.shift(14)
    # Rolling: primero shift(1) para no incluir el día actual, luego ventana de 28
    df["Sales_roll_28"] = grp.shift(1).rolling(window=28, min_periods=1).mean().reset_index(0, drop=True)

    # Imputación: NaN iniciales → 0 (media de Sales_norm por diseño)
    for col in ["Sales_lag_7", "Sales_lag_14", "Sales_roll_28"]:
        df[col] = df[col].fillna(0).astype(np.float32)

    return df


def build_sequences_ext(df: pd.DataFrame, seq_len: int = DEFAULT_SEQ_LEN) -> tuple:
    """
    Versión extendida de build_sequences que usa SEQ_FEATURE_COLS_EXT (14 features)
    en lugar de las 11 originales. Resto idéntico.
    """
    static_cols = STATIC_CAT_COLS + STATIC_NUM_COLS
    seq_parts, stat_parts, y_parts = [], [], []

    for _, grp in df.groupby("Store"):
        grp = grp.sort_values("Date")
        n = len(grp)
        if n <= seq_len:
            continue

        seq  = grp[SEQ_FEATURE_COLS_EXT].to_numpy(dtype=np.float32)
        stat = grp[static_cols].to_numpy(dtype=np.float32)
        y    = grp["Sales_norm"].to_numpy(dtype=np.float32)

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

# ============================================================
# Experimento 3: days_since_start movido a rama estática
# ============================================================

SEQ_FEATURE_COLS_EXT3 = [
    "Sales_norm",
    "Promo",
    "DayOfWeek_sin", "DayOfWeek_cos",
    "month_sin",     "month_cos",
    "week_sin",      "week_cos",
    "SchoolHoliday",
    "StateHoliday_enc",
    # "days_since_start" eliminado → ahora es feature estática
    "Sales_lag_7",
    "Sales_lag_14",
    "Sales_roll_28",
]

STATIC_NUM_COLS_EXT3 = [
    "CompetitionDistance_norm",
    "competition_age_years",
    "has_competition",
    "Promo2",
    "days_since_start_norm",
]


def add_days_since_start_norm(df: pd.DataFrame, train_mean: float, train_std: float) -> pd.DataFrame:
    """Normaliza days_since_start con media/std calculadas SOLO sobre train (sin leakage)."""
    df = df.copy()
    df["days_since_start_norm"] = ((df["days_since_start"] - train_mean) / train_std).astype(np.float32)
    return df


def build_sequences_ext3(df: pd.DataFrame, seq_len: int = DEFAULT_SEQ_LEN) -> tuple:
    """
    Versión exp3:
    - 13 features secuenciales (sin days_since_start)
    - 9 features estáticas (4 cat + 5 num: incluye days_since_start_norm)
    """
    static_cols = STATIC_CAT_COLS + STATIC_NUM_COLS_EXT3
    seq_parts, stat_parts, y_parts = [], [], []

    for _, grp in df.groupby("Store"):
        grp = grp.sort_values("Date")
        n = len(grp)
        if n <= seq_len:
            continue

        seq  = grp[SEQ_FEATURE_COLS_EXT3].to_numpy(dtype=np.float32)
        stat = grp[static_cols].to_numpy(dtype=np.float32)
        y    = grp["Sales_norm"].to_numpy(dtype=np.float32)

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