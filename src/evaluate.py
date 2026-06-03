"""
evaluate.py — Contrato común de evaluación. NO modificar sin consenso del equipo.

Esta es la fuente de verdad para:
  - Las fechas de corte del split temporal
  - Las tiendas objetivo que se evalúan
  - La función de R² que todos usan (misma implementación = métricas comparables)

El R² se calcula SIEMPRE sobre Sales en escala original (no normalizada), solo
sobre días con Open==1, solo para las tiendas de TARGET_STORES.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Constantes compartidas — importar desde aquí, no redefinir en otro módulo
# ---------------------------------------------------------------------------

TARGET_STORES = [1, 2, 3, 4, 5, 562, 682, 733, 769]

SPLIT_TRAIN_END  = pd.Timestamp("2014-09-30")
SPLIT_VAL_START  = pd.Timestamp("2014-10-01")
SPLIT_VAL_END    = pd.Timestamp("2014-12-31")
SPLIT_TEST_START = pd.Timestamp("2015-01-01")


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _inverse_transform(y_norm: np.ndarray, store_id: int, store_stats: dict) -> np.ndarray:
    """Desnormaliza: expm1(y_norm * std + mean)."""
    stats = store_stats[store_id]
    return np.expm1(y_norm * stats["std"] + stats["mean"])


def _per_store_arrays(
    df: pd.DataFrame,
    y_pred_norm: np.ndarray,
    store_stats: dict,
) -> dict:
    """
    Alinea y_pred_norm (producido por build_sequences) con las filas de df.

    build_sequences genera n_store - DEFAULT_SEQ_LEN predicciones por tienda,
    saltándose las primeras DEFAULT_SEQ_LEN filas (usadas como contexto).
    Este helper recorre las tiendas en el mismo orden que build_sequences
    (groupby "Store" con sort=True) para mantener el cursor sobre y_pred_norm.

    Devuelve {store_id: (y_true_euros, y_pred_euros)} solo para TARGET_STORES.
    """
    from src.preprocessing import DEFAULT_SEQ_LEN  # import local para evitar ciclo en módulo

    result = {}
    cursor = 0  # posición actual dentro de y_pred_norm

    for store_id, grp in df.groupby("Store"):
        grp_sorted = grp.sort_values("Date")
        n_preds = max(0, len(grp_sorted) - DEFAULT_SEQ_LEN)

        if store_id not in TARGET_STORES or n_preds == 0:
            cursor += n_preds
            continue

        # Filas con predicción (las primeras DEFAULT_SEQ_LEN son solo contexto)
        grp_target = grp_sorted.iloc[DEFAULT_SEQ_LEN:]
        y_true = grp_target["Sales"].values
        y_pred = _inverse_transform(
            y_pred_norm[cursor : cursor + n_preds], store_id, store_stats
        )
        result[store_id] = (y_true, y_pred)
        cursor += n_preds

    return result


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def r2_per_store(
    df: pd.DataFrame,
    y_pred_norm: np.ndarray,
    store_stats: dict,
) -> pd.Series:
    """R² individual para cada tienda de TARGET_STORES en el split de test."""
    pairs = _per_store_arrays(df, y_pred_norm, store_stats)
    return pd.Series(
        {sid: r2_score(yt, yp) for sid, (yt, yp) in pairs.items()}
    )


def r2_global(
    df: pd.DataFrame,
    y_pred_norm: np.ndarray,
    store_stats: dict,
) -> float:
    """R² pooled sobre todas las tiendas de TARGET_STORES en el split de test."""
    pairs = _per_store_arrays(df, y_pred_norm, store_stats)
    all_true = np.concatenate([yt for yt, _ in pairs.values()])
    all_pred = np.concatenate([yp for _, yp in pairs.values()])
    return float(r2_score(all_true, all_pred))


def evaluation_report(
    df: pd.DataFrame,
    y_pred_norm: np.ndarray,
    store_stats: dict,
) -> pd.DataFrame:
    """
    Tabla de resultados con R² por tienda + fila GLOBAL.

    Columnas: Store | n_samples | R2 | mean_sales_true | mean_sales_pred
    """
    pairs = _per_store_arrays(df, y_pred_norm, store_stats)

    rows = [
        {
            "Store":           sid,
            "n_samples":       len(yt),
            "R2":              r2_score(yt, yp),
            "mean_sales_true": yt.mean(),
            "mean_sales_pred": yp.mean(),
        }
        for sid, (yt, yp) in pairs.items()
    ]

    all_true = np.concatenate([yt for yt, _ in pairs.values()])
    all_pred = np.concatenate([yp for _, yp in pairs.values()])
    rows.append({
        "Store":           "GLOBAL",
        "n_samples":       len(all_true),
        "R2":              r2_score(all_true, all_pred),
        "mean_sales_true": all_true.mean(),
        "mean_sales_pred": all_pred.mean(),
    })

    report_df = pd.DataFrame(rows)
    print(report_df.to_string(index=False, float_format="{:.4f}".format))
    return report_df
