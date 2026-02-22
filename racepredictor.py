from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    model_name: str
    coefficient_a: float
    exponent_b: float
    rows_total: int
    rows_used: int
    rmse_seconds: float
    mae_seconds: float
    r2: float
    long_run_mae_seconds: float
    distance_cutoff: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the broad Peter Riegel model t = a * d^b from a CSV."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--distance-col",
        required=True,
        help="Column name containing run distance values.",
    )
    parser.add_argument(
        "--time-col",
        required=True,
        help="Column name containing run time values (e.g. 1:05:29).",
    )
    parser.add_argument(
        "--marathon-distance",
        required=True,
        type=float,
        help="Marathon distance to predict (in same units as CSV distance column).",
    )
    parser.add_argument(
        "--save-cleaned-rows",
        action="store_true",
        help="Save row-level fit outputs next to the CSV.",
    )
    parser.add_argument(
        "--race-date",
        help="Race date in YYYY-MM-DD; when provided, only runs on/before this date are used.",
    )
    parser.add_argument(
        "--date-col",
        help="Column name containing run date/time values (required when --race-date is used).",
    )
    return parser.parse_args()


def time_to_seconds(value: object) -> float:
    if pd.isna(value):
        return np.nan
    delta = pd.to_timedelta(str(value), errors="coerce")
    if pd.isna(delta):
        return np.nan
    return float(delta.total_seconds())


def seconds_to_hms(seconds: float) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_run_datetime(series: pd.Series) -> pd.Series:
    primary = pd.to_datetime(series, format="%m/%d/%y %H:%M", errors="coerce")
    if primary.notna().all():
        return primary
    fallback = pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    return primary.fillna(fallback)


def compute_regression_metrics(actual: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    mae = float(np.mean(np.abs(actual - pred)))
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, mae, r2


def fit_power_law(distance: pd.Series, time_seconds: pd.Series) -> tuple[float, float]:
    x = np.log(distance.to_numpy())
    y = np.log(time_seconds.to_numpy())
    b, ln_a = np.polyfit(x, y, 1)
    a = float(np.exp(ln_a))
    return a, float(b)


def load_and_clean(
    csv_path: Path,
    distance_col: str,
    time_col: str,
    race_date: pd.Timestamp | None = None,
    date_col: str | None = None,
) -> tuple[pd.DataFrame, int, int]:
    raw = pd.read_csv(csv_path)
    rows_total = len(raw)

    if distance_col not in raw.columns:
        raise ValueError(f"Distance column not found: {distance_col!r}")
    if time_col not in raw.columns:
        raise ValueError(f"Time column not found: {time_col!r}")

    cleaned = raw.copy()
    cleaned["distance"] = pd.to_numeric(cleaned[distance_col], errors="coerce")
    cleaned["time_seconds"] = cleaned[time_col].apply(time_to_seconds)

    if race_date is not None:
        if not date_col:
            raise ValueError("--date-col is required when --race-date is used.")
        if date_col not in cleaned.columns:
            raise ValueError(f"Date column not found: {date_col!r}")
        cleaned["run_datetime"] = parse_run_datetime(cleaned[date_col])
        cleaned = cleaned[cleaned["run_datetime"].notna()]
        cleaned = cleaned[cleaned["run_datetime"] <= race_date]

    cleaned = cleaned.dropna(subset=["distance", "time_seconds"])
    cleaned = cleaned[(cleaned["distance"] > 0) & (cleaned["time_seconds"] > 0)]
    cleaned["pace_seconds_per_unit"] = cleaned["time_seconds"] / cleaned["distance"]

    rows_used = len(cleaned)
    if rows_used < 2:
        raise ValueError("Need at least 2 valid rows to fit the Riegel model.")
    return cleaned, rows_total, rows_used


def pick_long_run_eval_set(cleaned: pd.DataFrame) -> pd.DataFrame:
    for q in (0.90, 0.85, 0.80, 0.75):
        cutoff = float(cleaned["distance"].quantile(q))
        eval_set = cleaned[cleaned["distance"] >= cutoff]
        if len(eval_set) >= 8:
            return eval_set
    return cleaned.nlargest(min(len(cleaned), 8), "distance")


def fit_candidate_model(
    model_name: str,
    training_df: pd.DataFrame,
    full_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    rows_total: int,
    rows_used: int,
    distance_cutoff: float | None,
) -> tuple[FitResult, pd.DataFrame]:
    a, b = fit_power_law(training_df["distance"], training_df["time_seconds"])
    fitted = full_df.copy()
    fitted["predicted_seconds"] = a * np.power(fitted["distance"], b)
    fitted["residual_seconds"] = fitted["time_seconds"] - fitted["predicted_seconds"]

    actual = fitted["time_seconds"].to_numpy()
    pred = fitted["predicted_seconds"].to_numpy()
    rmse, mae, r2 = compute_regression_metrics(actual, pred)

    eval_pred = a * np.power(eval_df["distance"].to_numpy(), b)
    long_run_mae = float(np.mean(np.abs(eval_df["time_seconds"].to_numpy() - eval_pred)))

    return FitResult(
        model_name=model_name,
        coefficient_a=a,
        exponent_b=b,
        rows_total=rows_total,
        rows_used=rows_used,
        rmse_seconds=rmse,
        mae_seconds=mae,
        r2=r2,
        long_run_mae_seconds=long_run_mae,
        distance_cutoff=distance_cutoff,
    ), fitted


def fit_riegel_power_law(
    csv_path: Path,
    distance_col: str,
    time_col: str,
    race_date: pd.Timestamp | None = None,
    date_col: str | None = None,
) -> tuple[FitResult, pd.DataFrame]:
    cleaned, rows_total, rows_used = load_and_clean(
        csv_path, distance_col, time_col, race_date=race_date, date_col=date_col
    )
    eval_df = pick_long_run_eval_set(cleaned)

    candidates: list[tuple[FitResult, pd.DataFrame]] = []

    all_runs_result = fit_candidate_model(
        model_name="all-runs",
        training_df=cleaned,
        full_df=cleaned,
        eval_df=eval_df,
        rows_total=rows_total,
        rows_used=rows_used,
        distance_cutoff=None,
    )
    candidates.append(all_runs_result)

    for q in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85):
        cutoff = float(cleaned["distance"].quantile(q))
        training_df = cleaned[cleaned["distance"] >= cutoff]
        if len(training_df) < 8:
            continue
        candidates.append(
            fit_candidate_model(
                model_name=f"long-run-focus q={q:.2f}",
                training_df=training_df,
                full_df=cleaned,
                eval_df=eval_df,
                rows_total=rows_total,
                rows_used=rows_used,
                distance_cutoff=cutoff,
            )
        )

    best_result, best_fitted = min(candidates, key=lambda c: c[0].long_run_mae_seconds)
    return best_result, best_fitted


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if args.marathon_distance <= 0:
        raise ValueError("Marathon distance must be > 0.")
    race_date = None
    if args.race_date:
        race_date = pd.to_datetime(args.race_date, format="%Y-%m-%d", errors="raise")

    fit_result, cleaned = fit_riegel_power_law(
        csv_path=csv_path,
        distance_col=args.distance_col,
        time_col=args.time_col,
        race_date=race_date,
        date_col=args.date_col,
    )

    print("Marathon-focused Riegel fit completed from CSV data.")
    print(f"CSV: {csv_path}")
    print(f"Rows total: {fit_result.rows_total}")
    print(f"Rows used: {fit_result.rows_used}")
    if race_date is not None:
        print(f"Race date filter: runs on/before {race_date.date()}")
    print(f"Selected model: {fit_result.model_name}")
    if fit_result.distance_cutoff is not None:
        print(f"Distance cutoff for training rows: {fit_result.distance_cutoff:.4f}")
    print()
    print("Fitted equation (from your CSV):")
    print(f"t = a * d^b")
    print(f"a = {fit_result.coefficient_a:.6f}")
    print(f"b = {fit_result.exponent_b:.6f}")
    print(f"t = {fit_result.coefficient_a:.6f} * d^{fit_result.exponent_b:.6f}")
    print()
    print("Fit quality on CSV rows:")
    print(f"RMSE (sec): {fit_result.rmse_seconds:.3f}")
    print(f"MAE  (sec): {fit_result.mae_seconds:.3f}")
    print(f"R^2       : {fit_result.r2:.5f}")
    print(f"Long-run MAE (sec): {fit_result.long_run_mae_seconds:.3f}")
    print()
    marathon_time_seconds = fit_result.coefficient_a * (
        args.marathon_distance ** fit_result.exponent_b
    )
    marathon_time_hms = seconds_to_hms(marathon_time_seconds)

    print("Marathon prediction:")
    print(f"Distance: {args.marathon_distance}")
    print(f"Seconds : {marathon_time_seconds:.3f}")
    print(f"HMS     : {marathon_time_hms}")

    if args.save_cleaned_rows:
        # Save detailed row-level outputs next to the CSV for traceability.
        output_path = csv_path.with_name(f"{csv_path.stem}_riegel_fit_rows.csv")
        cleaned_out = cleaned.copy()
        cleaned_out["actual_hms"] = cleaned_out["time_seconds"].apply(seconds_to_hms)
        cleaned_out["predicted_hms"] = cleaned_out["predicted_seconds"].apply(seconds_to_hms)
        cleaned_out.to_csv(output_path, index=False)
        print()
        print(f"Saved row-level fit output: {output_path}")


if __name__ == "__main__":
    main()
