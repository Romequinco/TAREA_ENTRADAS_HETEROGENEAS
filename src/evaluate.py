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

SPLIT_TRAIN_END  = pd.Timestamp("2014-09-30")   # último día de entrenamiento
SPLIT_VAL_START  = pd.Timestamp("2014-10-01")   # inicio de validación interna
SPLIT_VAL_END    = pd.Timestamp("2014-12-31")   # fin de validación (incluye Navidad)
SPLIT_TEST_START = pd.Timestamp("2015-01-01")   # inicio del test evaluado por el profesor


# ---------------------------------------------------------------------------
# Funciones de evaluación
# ---------------------------------------------------------------------------

def _inverse_transform(y_norm: np.ndarray, store_id: int, store_stats: dict) -> np.ndarray:
    """
    Desnormaliza predicciones: expm1(y_norm * std + mean).

    Asume que la normalización aplicada fue:
        y_norm = (log1p(Sales) - mean_store) / std_store

    Parameters
    ----------
    y_norm : np.ndarray, shape (n,)
        Valores normalizados a deshacer.
    store_id : int
        ID de la tienda (clave en store_stats).
    store_stats : dict
        {store_id: {'mean': float, 'std': float}} generado por
        preprocessing.build_store_normalizer sobre el split de train.

    Returns
    -------
    np.ndarray
        Sales predichas en escala original (euros).
    """
    stats = store_stats[store_id]
    return np.expm1(y_norm * stats["std"] + stats["mean"])


def r2_per_store(
    df: pd.DataFrame,
    y_pred_norm: np.ndarray,
    store_stats: dict,
) -> pd.Series:
    """
    Calcula R² individual para cada tienda de TARGET_STORES en el split de test.

    Protocolo (NO alterar):
      1. Filtra df a Date >= SPLIT_TEST_START y Open == 1.
      2. Para cada tienda en TARGET_STORES presente en df:
         a. Extrae índices de esas filas en df filtrado.
         b. Recupera Sales_true = df["Sales"].
         c. Recupera Sales_pred = _inverse_transform(y_pred_norm[índices], store_id, store_stats).
         d. Calcula r2_score(Sales_true, Sales_pred).

    Parameters
    ----------
    df : pd.DataFrame
        Debe contener columnas [Store, Date, Open, Sales].
        El orden de filas debe coincidir exactamente con y_pred_norm.
    y_pred_norm : np.ndarray, shape (n,)
        Salida del modelo en espacio normalizado, mismo orden que df.
    store_stats : dict
        {store_id: {'mean': float, 'std': float}} de preprocessing.build_store_normalizer.

    Returns
    -------
    pd.Series
        Index = store_id (int), values = R² (float).
        Solo incluye tiendas de TARGET_STORES presentes en df.
    """
    raise NotImplementedError


def r2_global(
    df: pd.DataFrame,
    y_pred_norm: np.ndarray,
    store_stats: dict,
) -> float:
    """
    R² agregado sobre todas las tiendas de TARGET_STORES en el split de test.

    Concatena Sales_true y Sales_pred de todas las tiendas objetivo y aplica
    un único r2_score sobre el vector concatenado (pooled R²).

    Parameters
    ----------
    df : pd.DataFrame
        Mismo contrato que r2_per_store.
    y_pred_norm : np.ndarray, shape (n,)
        Mismo contrato que r2_per_store.
    store_stats : dict
        Mismo contrato que r2_per_store.

    Returns
    -------
    float
        R² global (pooled).
    """
    raise NotImplementedError


def evaluation_report(
    df: pd.DataFrame,
    y_pred_norm: np.ndarray,
    store_stats: dict,
) -> pd.DataFrame:
    """
    Imprime y devuelve la tabla de resultados con R² por tienda y global.

    Columnas del DataFrame devuelto:
        Store           : int o 'GLOBAL'
        n_samples       : número de días en test (Open==1)
        R2              : R² de esa tienda (o pooled si Store=='GLOBAL')
        mean_sales_true : media de Sales reales en test
        mean_sales_pred : media de Sales predichas en test

    La última fila tiene Store='GLOBAL' con el pooled R² de r2_global().

    Parameters
    ----------
    df : pd.DataFrame
        Mismo contrato que r2_per_store.
    y_pred_norm : np.ndarray, shape (n,)
        Mismo contrato que r2_per_store.
    store_stats : dict
        Mismo contrato que r2_per_store.

    Returns
    -------
    pd.DataFrame
        Tabla de resultados (también impresa por stdout con formato legible).
    """
    raise NotImplementedError
