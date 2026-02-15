import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: scikit-learn. Install with:\n"
        "python3 -m pip install scikit-learn matplotlib pandas numpy"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare race time prediction models from a CSV file."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--target",
        default="Time",
        help="Target column to predict (default: Time).",
    )
    parser.add_argument(
        "--distance-col",
        default="Distance",
        help="Distance feature column used for the power-law model (default: Distance).",
    )
    parser.add_argument(
        "--predict-distance",
        type=float,
        default=26.2,
        help="Distance for race-time forecast (default: 26.2).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for plots and summary CSV files (default: outputs).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Cross-validation folds (default: 3, lower is faster).",
    )
    return parser.parse_args()


def format_hms(seconds: float) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def duration_to_seconds(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan

    parts = text.split(":")
    try:
        if len(parts) == 3:
            h = float(parts[0])
            m = float(parts[1])
            s = float(parts[2])
            return h * 3600 + m * 60 + s
        if len(parts) == 2:
            m = float(parts[0])
            s = float(parts[1])
            return m * 60 + s
        return float(text.replace(",", ""))
    except ValueError:
        return np.nan


def coerce_object_column(series: pd.Series, col_name: str) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return series

    as_str = non_null.astype(str).str.strip()
    cleaned = as_str.str.replace(",", "", regex=False)

    # Parse date-like columns.
    if "date" in col_name.lower():
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        if parsed.notna().sum() >= max(10, int(0.5 * len(non_null))):
            return parsed

    # Parse duration-like strings such as HH:MM:SS or MM:SS(.s)
    duration_pattern = r"^\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?$"
    duration_ratio = float(as_str.str.match(duration_pattern).mean())
    if duration_ratio >= 0.6:
        return series.map(duration_to_seconds)

    # Parse numeric-like strings with commas.
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if numeric.notna().sum() >= max(10, int(0.6 * len(non_null))):
        return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")

    return series


def load_data(csv_path: Path, target_col: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    # Parse object columns into usable numeric/time values where possible.
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == "object":
            df[col] = coerce_object_column(df[col], col)

    # Expand datetime columns into numeric features and drop raw datetime.
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for col in datetime_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_weekday"] = df[col].dt.weekday
        df[f"{col}_hour"] = df[col].dt.hour
        df[f"{col}_minute"] = df[col].dt.minute
    df = df.drop(columns=list(datetime_cols))

    # Ensure target is numeric (handle both numeric and time-like text)
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        if pd.api.types.is_string_dtype(df[target_col]) or df[target_col].dtype == "object":
            df[target_col] = df[target_col].map(duration_to_seconds)
        else:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    df = df.dropna(subset=[target_col])

    if len(df) < 10:
        raise ValueError("Need at least 10 valid rows after cleaning for reliable model evaluation.")

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE_pct": mape}


def evaluate_models(
    X: pd.DataFrame,
    y: np.ndarray,
    distance_col: str,
    random_state: int,
    cv_folds: int = 3,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Pipeline]]:
    preprocessor = build_preprocessor(X)

    models: dict[str, Pipeline] = {
        "LinearRegression": Pipeline(
            steps=[("prep", preprocessor), ("model", LinearRegression())]
        ),
        "RidgeCV": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", RidgeCV(alphas=np.logspace(-3, 3, 30))),
            ]
        ),
        "GradientBoosting": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    # Power-law baseline in log-log space if distance exists
    if distance_col in X.columns:
        positive_mask = (X[distance_col] > 0) & (y > 0)
        if positive_mask.sum() >= 10:
            power_pipe = Pipeline(
                steps=[
                    (
                        "selector",
                        FunctionTransformer(
                            lambda d: np.log(
                                np.clip(d[[distance_col]].to_numpy(dtype=float), 1e-6, None)
                            ),
                            feature_names_out="one-to-one",
                            validate=False,
                        ),
                    ),
                    ("model", LinearRegression()),
                ]
            )
            models["PowerLaw(DistanceOnly)"] = power_pipe

    fold_count = max(2, min(cv_folds, len(X)))
    cv = KFold(n_splits=fold_count, shuffle=True, random_state=random_state)

    rows: list[dict] = []
    preds: dict[str, np.ndarray] = {}

    for name, model in models.items():
        if "PowerLaw" in name:
            y_log_pred = cross_val_predict(model, X, np.log(np.clip(y, 1e-6, None)), cv=cv)
            y_pred = np.exp(y_log_pred)
        else:
            y_pred = cross_val_predict(model, X, y, cv=cv)

        model_metrics = metrics(y, y_pred)
        rows.append({"Model": name, **model_metrics})
        preds[name] = y_pred

    results = pd.DataFrame(rows).sort_values(by="RMSE", ascending=True).reset_index(drop=True)

    return results, preds, models


def fit_best_model(
    model_name: str,
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: np.ndarray,
) -> Pipeline:
    model = models[model_name]

    if "PowerLaw" in model_name:
        model.fit(X, np.log(np.clip(y, 1e-6, None)))
    else:
        model.fit(X, y)

    return model


def predict_for_distance(
    model_name: str,
    model: Pipeline,
    X: pd.DataFrame,
    distance_col: str,
    predict_distance: float,
    y: np.ndarray,
) -> float | None:
    if distance_col not in X.columns:
        return None

    template = X.median(numeric_only=True).to_dict()

    # Fill all columns in the same order expected by model.
    row = {}
    for col in X.columns:
        if col == distance_col:
            row[col] = predict_distance
        elif col in template:
            row[col] = template[col]
        else:
            # For non-numeric columns, use most frequent category from training data.
            row[col] = X[col].mode(dropna=True).iloc[0] if not X[col].mode(dropna=True).empty else "unknown"

    X_pred = pd.DataFrame([row], columns=X.columns)

    if "PowerLaw" in model_name:
        pred = float(np.exp(model.predict(X_pred))[0])
    else:
        pred = float(model.predict(X_pred)[0])

    # Avoid pathological negative predictions for regressors
    return max(pred, 0.0)


def save_plots(
    y: np.ndarray,
    best_name: str,
    best_pred: np.ndarray,
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Model comparison chart
    plt.figure(figsize=(10, 5))
    plt.bar(results["Model"], results["RMSE"], color="#2563eb")
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("RMSE (seconds)")
    plt.title("Model Comparison (Lower is Better)")
    plt.tight_layout()
    plt.savefig(output_dir / "model_rmse_comparison.png", dpi=150)
    plt.close()

    # 2) Actual vs predicted
    lims = [min(y.min(), best_pred.min()), max(y.max(), best_pred.max())]
    plt.figure(figsize=(6, 6))
    plt.scatter(y, best_pred, alpha=0.75, edgecolor="k", linewidth=0.3)
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual Time (seconds)")
    plt.ylabel("Predicted Time (seconds)")
    plt.title(f"Actual vs Predicted ({best_name})")
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # 3) Residual distribution
    residuals = y - best_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=25, color="#16a34a", alpha=0.85, edgecolor="white")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Residual (Actual - Predicted, seconds)")
    plt.ylabel("Count")
    plt.title(f"Residual Distribution ({best_name})")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_distribution.png", dpi=150)
    plt.close()


def run_pipeline(
    csv_path: Path,
    target: str = "Time",
    distance_col: str = "Distance",
    predict_distance: float = 26.2,
    output_dir: Path = Path("outputs"),
    random_state: int = 42,
    cv_folds: int = 3,
) -> dict:
    csv_path = Path(csv_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    df = load_data(csv_path, target)
    X = df.drop(columns=[target])
    y = df[target].to_numpy(dtype=float)

    results, preds, models = evaluate_models(
        X=X,
        y=y,
        distance_col=distance_col,
        random_state=random_state,
        cv_folds=cv_folds,
    )

    best_model_name = results.iloc[0]["Model"]
    best_pred = preds[best_model_name]
    best_metrics = results.iloc[0].to_dict()

    best_model = fit_best_model(best_model_name, models, X, y)
    race_time_prediction = predict_for_distance(
        model_name=best_model_name,
        model=best_model,
        X=X,
        distance_col=distance_col,
        predict_distance=predict_distance,
        y=y,
    )

    save_plots(
        y=y,
        best_name=best_model_name,
        best_pred=best_pred,
        results=results,
        output_dir=output_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "model_metrics.csv"
    preds_path = output_dir / "best_model_predictions.csv"
    results.to_csv(metrics_path, index=False)
    pd.DataFrame({"actual": y, "predicted": best_pred}).to_csv(preds_path, index=False)

    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "race_time_prediction": race_time_prediction,
        "prediction_hms": format_hms(race_time_prediction) if race_time_prediction is not None else None,
        "output_dir": output_dir,
        "files": {
            "metrics_csv": metrics_path,
            "predictions_csv": preds_path,
            "rmse_plot": output_dir / "model_rmse_comparison.png",
            "actual_vs_pred_plot": output_dir / "actual_vs_predicted.png",
            "residual_plot": output_dir / "residual_distribution.png",
        },
    }


def main() -> None:
    args = parse_args()

    run = run_pipeline(
        csv_path=Path(args.csv),
        target=args.target,
        distance_col=args.distance_col,
        predict_distance=args.predict_distance,
        output_dir=Path(args.output_dir),
        random_state=args.random_state,
        cv_folds=args.cv_folds,
    )
    results = run["results"]
    best_model_name = run["best_model_name"]
    best_metrics = run["best_metrics"]
    race_time_prediction = run["race_time_prediction"]
    output_dir = run["output_dir"]

    print("\n===== MODEL RANKING =====")
    print(results.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print("\n===== BEST MODEL =====")
    print(f"Model: {best_model_name}")
    print(
        f"RMSE: {best_metrics['RMSE']:.2f}s | MAE: {best_metrics['MAE']:.2f}s | "
        f"R²: {best_metrics['R2']:.4f} | MAPE: {best_metrics['MAPE_pct']:.2f}%"
    )

    if race_time_prediction is not None:
        print("\n===== RACE FORECAST =====")
        print(
            f"Predicted time for distance {args.predict_distance} using median-profile inputs: "
            f"{race_time_prediction:.2f} sec ({run['prediction_hms']})"
        )
    else:
        print(
            f"\nDistance column '{args.distance_col}' not found, so distance-based forecast was skipped."
        )

    print("\nOutputs written to:")
    print(output_dir)
    print("- model_metrics.csv")
    print("- best_model_predictions.csv")
    print("- model_rmse_comparison.png")
    print("- actual_vs_predicted.png")
    print("- residual_distribution.png")


if __name__ == "__main__":
    main()
