import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluate import TARGET_STORES
from src.preprocessing import DEFAULT_SEQ_LEN


def _display(obj, title=None):
    if title:
        print(f"\n{title}")
    try:
        display(obj)
    except NameError:
        print(obj)


def _safe_cv(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    mean = values.mean()
    return np.nan if pd.isna(mean) or mean == 0 else values.std() / mean


def _promo_ratio(df):
    if df.empty or not {"Promo", "Sales"}.issubset(df.columns):
        return np.nan
    promo = pd.to_numeric(df["Promo"], errors="coerce").fillna(0)
    sales = pd.to_numeric(df["Sales"], errors="coerce")
    promo_mean = sales[promo > 0].mean()
    non_promo_mean = sales[promo <= 0].mean()
    if pd.isna(promo_mean) or pd.isna(non_promo_mean) or non_promo_mean == 0:
        return np.nan
    return promo_mean / non_promo_mean


def _split_summary(name, df):
    sales = pd.to_numeric(df["Sales"], errors="coerce") if not df.empty else pd.Series(dtype=float)
    return {
        "split": name,
        "n_obs": len(df),
        "date_range": "n/a" if df.empty else f"{df['Date'].min().date()} -> {df['Date'].max().date()}",
        "mean_sales": sales.mean(),
        "median_sales": sales.median(),
        "std_sales": sales.std(),
        "cv": _safe_cv(sales),
        "promo_pct": 100 * pd.to_numeric(df["Promo"], errors="coerce").fillna(0).gt(0).mean() if "Promo" in df else np.nan,
    }


def build_prediction_frame(
    df_test: pd.DataFrame,
    y_pred_norm: np.ndarray,
    store_stats: dict,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> pd.DataFrame:
    y_pred_norm = np.asarray(y_pred_norm, dtype=float).reshape(-1)
    rows = []
    cursor = 0

    for store_id, grp in df_test.groupby("Store", sort=True):
        grp = grp.sort_values("Date").copy()
        n_preds = max(0, len(grp) - seq_len)
        if n_preds == 0:
            continue

        if cursor + n_preds > len(y_pred_norm):
            raise ValueError("No hay suficientes predicciones para alinear df_test.")

        stats = store_stats[int(store_id)]
        pred = y_pred_norm[cursor:cursor + n_preds]
        pred = np.expm1(pred * stats["std"] + stats["mean"])
        cursor += n_preds

        aligned = grp.iloc[seq_len:].copy()
        aligned["predicted_Sales"] = pred
        rows.append(aligned)

    pred_df = pd.concat(rows, ignore_index=True)
    pred_df["residual"] = pred_df["Sales"] - pred_df["predicted_Sales"]
    pred_df["abs_error"] = pred_df["residual"].abs()
    pred_df["pct_error"] = 100 * pred_df["residual"] / pred_df["Sales"].replace(0, np.nan)
    if cursor != len(y_pred_norm):
        raise ValueError("Sobran predicciones tras alinear df_test.")
    return pred_df


def diagnose_store(
    store_id: int,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    y_pred_norm: np.ndarray,
    store_stats: dict,
    report_df: pd.DataFrame | None = None,
    target_stores=None,
    seq_len: int = DEFAULT_SEQ_LEN,
    show_plots: bool = True,
):
    target_stores = TARGET_STORES if target_stores is None else list(target_stores)
    pred_df = build_prediction_frame(df_test, y_pred_norm, store_stats, seq_len=seq_len)

    store_train = df_train[df_train["Store"] == store_id].sort_values("Date").copy()
    store_val = df_val[df_val["Store"] == store_id].sort_values("Date").copy()
    store_test = df_test[df_test["Store"] == store_id].sort_values("Date").copy()
    diag_df = pred_df[pred_df["Store"] == store_id].sort_values("Date").copy()
    keep_cols = ["Date", "Sales", "predicted_Sales", "residual", "abs_error", "pct_error"]
    keep_cols += [col for col in ["Promo", "DayOfWeek", "SchoolHoliday", "StateHoliday"] if col in diag_df.columns]
    diag_df = diag_df[keep_cols]

    if store_train.empty and store_val.empty and store_test.empty:
        raise ValueError(f"Store {store_id} no aparece en los splits.")
    if diag_df.empty:
        raise ValueError(f"Store {store_id} no tiene filas evaluables después de aplicar seq_len={seq_len}.")

    summary_df = pd.DataFrame(
        [
            _split_summary("train", store_train),
            _split_summary("validation", store_val),
            _split_summary("test", store_test),
        ]
    )

    meta_cols = [
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "Promo2",
        "competition_age_years",
    ]
    meta_source = next((df for df in [store_train, store_val, store_test] if not df.empty), pd.DataFrame())
    metadata_df = (
        meta_source[meta_cols]
        .iloc[[0]]
        .T.rename(columns={meta_source.index[0]: "value"})
        .reset_index()
        .rename(columns={"index": "feature"})
        if not meta_source.empty else pd.DataFrame(columns=["feature", "value"])
    )

    y_true = diag_df["Sales"].to_numpy(dtype=float)
    y_pred = diag_df["predicted_Sales"].to_numpy(dtype=float)
    baseline_mean = float(store_train["Sales"].mean())
    baseline = np.full_like(y_true, baseline_mean)

    metrics_df = pd.DataFrame(
        [
            {"metric": "R2", "model": r2_score(y_true, y_pred), "baseline": r2_score(y_true, baseline)},
            {"metric": "MAE", "model": mean_absolute_error(y_true, y_pred), "baseline": mean_absolute_error(y_true, baseline)},
            {"metric": "RMSE", "model": np.sqrt(mean_squared_error(y_true, y_pred)), "baseline": np.sqrt(mean_squared_error(y_true, baseline))},
        ]
    )

    report_clean = pd.DataFrame(columns=["Store", "R2"])
    if report_df is not None and not report_df.empty and {"Store", "R2"}.issubset(report_df.columns):
        report_clean = report_df.copy()
        report_clean["Store"] = pd.to_numeric(report_clean["Store"], errors="coerce")
        report_clean = report_clean.dropna(subset=["Store"]).assign(Store=lambda d: d["Store"].astype(int))

    comparison = (
        pred_df[pred_df["Store"].isin(target_stores)]
        .groupby("Store")
        .agg(
            n_obs=("Sales", "size"),
            mean_sales=("Sales", "mean"),
            cv=("Sales", _safe_cv),
            mean_abs_error=("abs_error", "mean"),
        )
        .reset_index()
        .merge(report_clean[["Store", "R2"]], on="Store", how="left")
    )

    peers = comparison[comparison["Store"] != store_id]
    comparison_rows = [comparison[comparison["Store"] == store_id].assign(label=f"Store {store_id}")]
    if not peers.empty:
        comparison_rows.append(pd.DataFrame([{
            "label": "Average peers",
            "Store": np.nan,
            "n_obs": peers["n_obs"].mean(),
            "mean_sales": peers["mean_sales"].mean(),
            "cv": peers["cv"].mean(),
            "mean_abs_error": peers["mean_abs_error"].mean(),
            "R2": peers["R2"].mean(),
        }]))
        if peers["R2"].notna().any():
            comparison_rows.append(peers.nlargest(1, "R2").assign(label="Best peer"))
            comparison_rows.append(peers.nsmallest(1, "R2").assign(label="Worst peer"))
    comparison_df = pd.concat(comparison_rows, ignore_index=True)

    mean_residual = float(diag_df["residual"].mean())
    pred_std_ratio = float(np.std(y_pred) / max(np.std(y_true), 1e-8))
    train_mean = float(store_train["Sales"].mean())
    test_mean = float(diag_df["Sales"].mean())
    mean_shift_pct = 100 * (test_mean - train_mean) / train_mean if train_mean else np.nan
    train_cv = _safe_cv(store_train["Sales"])
    test_cv = _safe_cv(diag_df["Sales"])
    train_promo_ratio = _promo_ratio(store_train)
    test_promo_ratio = _promo_ratio(diag_df)

    checks_df = pd.DataFrame(
        [
            {"check": "Train mean sales", "value": train_mean},
            {"check": "Test mean sales", "value": test_mean},
            {"check": "Mean shift train->test (%)", "value": mean_shift_pct},
            {"check": "Train CV", "value": train_cv},
            {"check": "Test CV", "value": test_cv},
            {"check": "Train promo lift ratio", "value": train_promo_ratio},
            {"check": "Test promo lift ratio", "value": test_promo_ratio},
            {"check": "Mean residual", "value": mean_residual},
            {"check": "Pred std / true std", "value": pred_std_ratio},
        ]
    )

    findings = []
    suggestions = []

    if abs(mean_shift_pct) >= 15:
        findings.append(f"hay cambio train-test en ventas medias ({mean_shift_pct:+.1f}%)")
        suggestions.append("calibración por tienda")
    if pd.notna(test_cv) and pd.notna(train_cv) and test_cv > 1.15 * train_cv:
        findings.append("el test es bastante más volátil que el train")
        suggestions.append("ponderar más la tienda 769 o afinar por tienda")
    if mean_residual > 0.05 * y_true.mean():
        findings.append("el modelo tiende a infrapredecir")
        suggestions.append("corrección de sesgo por tienda")
    elif mean_residual < -0.05 * y_true.mean():
        findings.append("el modelo tiende a sobrepredecir")
        suggestions.append("corrección de sesgo por tienda")
    if pred_std_ratio < 0.85:
        findings.append("las predicciones son demasiado suaves frente a la serie real")
        suggestions.append("modelo más sensible a picos")

    peak_mask = diag_df["Sales"] >= diag_df["Sales"].quantile(0.90)
    if diag_df.loc[peak_mask, "abs_error"].mean() > 1.25 * diag_df["abs_error"].mean():
        findings.append("los días pico concentran más error que el resto")
        suggestions.append("features o arquitectura que capturen mejor picos")

    if not findings:
        findings.append("no hay una causa única; parece mezcla de shift y dinámica específica de la tienda")
        suggestions.extend(["calibración por tienda", "fine-tuning por tienda"])

    suggestions = list(dict.fromkeys(suggestions))

    _display(summary_df.round(4), f"Store {store_id} split summary")
    _display(metadata_df, f"Store {store_id} metadata")
    _display(diag_df.head(10).round(4), f"Store {store_id} diagnostic dataframe")
    _display(comparison_df.round(4), f"Store {store_id} vs peers")
    _display(metrics_df.round(4), f"Store {store_id} model vs baseline")
    _display(checks_df.round(4), f"Store {store_id} diagnostic checks")

    print("\nConclusion")
    print(
        f"Store {store_id} destaca porque " + "; ".join(findings[:4]) +
        ". Las pruebas siguientes con más sentido son: " + "; ".join(suggestions[:4]) + "."
    )

    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(store_train["Date"], store_train["Sales"], label="Train")
        axes[0, 0].plot(store_val["Date"], store_val["Sales"], label="Validation")
        axes[0, 0].plot(store_test["Date"], store_test["Sales"], label="Test")
        axes[0, 0].set_title(f"Store {store_id}: sales by split")
        axes[0, 0].legend()

        axes[0, 1].plot(diag_df["Date"], diag_df["Sales"], label="True")
        axes[0, 1].plot(diag_df["Date"], diag_df["predicted_Sales"], label="Predicted")
        axes[0, 1].set_title(f"Store {store_id}: true vs predicted")
        axes[0, 1].legend()

        axes[1, 0].plot(diag_df["Date"], diag_df["residual"], color="tab:purple")
        axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[1, 0].set_title(f"Store {store_id}: residuals")

        axes[1, 1].scatter(diag_df["Sales"], diag_df["predicted_Sales"], alpha=0.6)
        low = min(diag_df["Sales"].min(), diag_df["predicted_Sales"].min())
        high = max(diag_df["Sales"].max(), diag_df["predicted_Sales"].max())
        axes[1, 1].plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        axes[1, 1].set_title(f"Store {store_id}: true vs predicted scatter")
        axes[1, 1].set_xlabel("True")
        axes[1, 1].set_ylabel("Predicted")

        plt.tight_layout()
        plt.show()

    return {
        "summary": summary_df,
        "metadata": metadata_df,
        "diagnostic": diag_df,
        "comparison": comparison_df,
        "metrics": metrics_df,
        "checks": checks_df,
        "findings": findings,
        "suggestions": suggestions,
        "prediction_frame": pred_df,
    }
