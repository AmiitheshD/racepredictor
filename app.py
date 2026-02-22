from __future__ import annotations

import os
import traceback
import uuid
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

# Render/Linux containers can have non-writable home config paths.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from racepredictor import fit_riegel_power_law, seconds_to_hms

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"csv"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}, 200


@app.post("/predict")
def predict():
    file = request.files.get("csv_file")
    time_col = request.form.get("time_col", "Time").strip() or "Time"
    distance_col = request.form.get("distance_col", "Distance").strip() or "Distance"
    date_col = request.form.get("date_col", "Date").strip() or "Date"
    race_date = request.form.get("race_date", "").strip()

    try:
        marathon_distance = float(request.form.get("marathon_distance", "26.2188"))
    except ValueError:
        return render_template("index.html", error="Marathon distance must be a number.")

    if not file or not file.filename:
        return render_template("index.html", error="Please upload a CSV file.")

    if not allowed_file(file.filename):
        return render_template("index.html", error="Only .csv files are supported.")

    run_id = uuid.uuid4().hex[:12]
    filename = secure_filename(file.filename)
    upload_path = UPLOAD_DIR / f"{run_id}_{filename}"
    file.save(upload_path)

    try:
        fit_result, _ = fit_riegel_power_law(
            csv_path=upload_path,
            distance_col=distance_col,
            time_col=time_col,
            race_date=race_date if race_date else None,
            date_col=date_col if race_date else None,
        )
        marathon_seconds = fit_result.coefficient_a * (
            marathon_distance ** fit_result.exponent_b
        )
        marathon_hms = seconds_to_hms(marathon_seconds)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        return render_template("index.html", error=f"Could not run prediction: {exc}")
    finally:
        upload_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        success=True,
        marathon_hms=marathon_hms,
        marathon_seconds=marathon_seconds,
        marathon_distance=marathon_distance,
        model_name=fit_result.model_name,
        coefficient_a=fit_result.coefficient_a,
        exponent_b=fit_result.exponent_b,
        rows_total=fit_result.rows_total,
        rows_used=fit_result.rows_used,
        rmse_seconds=fit_result.rmse_seconds,
        mae_seconds=fit_result.mae_seconds,
        r2=fit_result.r2,
        long_run_mae_seconds=fit_result.long_run_mae_seconds,
        distance_cutoff=fit_result.distance_cutoff,
        race_date=race_date,
    )


@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception):
    if isinstance(exc, HTTPException):
        return render_template("index.html", error=str(exc)), exc.code
    app.logger.error("Unhandled server error: %s\n%s", exc, traceback.format_exc())
    return render_template("index.html", error=f"Server error: {exc}"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
