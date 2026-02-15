from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

# Render/Linux containers can have non-writable home config paths.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from race_predictor import run_pipeline

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_ROOT = BASE_DIR / "web_outputs"
ALLOWED_EXTENSIONS = {"csv"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    file = request.files.get("csv_file")
    target = request.form.get("target", "Time").strip() or "Time"
    distance_col = request.form.get("distance_col", "Distance").strip() or "Distance"

    try:
        predict_distance = float(request.form.get("predict_distance", "26.2"))
    except ValueError:
        return render_template("index.html", error="Predict distance must be a number.")

    if not file or not file.filename:
        return render_template("index.html", error="Please upload a CSV file.")

    if not allowed_file(file.filename):
        return render_template("index.html", error="Only .csv files are supported.")

    run_id = uuid.uuid4().hex[:12]
    filename = secure_filename(file.filename)
    upload_path = UPLOAD_DIR / f"{run_id}_{filename}"
    file.save(upload_path)

    run_output_dir = OUTPUT_ROOT / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_pipeline(
            csv_path=upload_path,
            target=target,
            distance_col=distance_col,
            predict_distance=predict_distance,
            output_dir=run_output_dir,
        )
    except Exception as exc:
        shutil.rmtree(run_output_dir, ignore_errors=True)
        upload_path.unlink(missing_ok=True)
        return render_template("index.html", error=f"Could not run prediction: {exc}")

    chart_urls = {
        "rmse": url_for("run_file", run_id=run_id, filename="model_rmse_comparison.png"),
        "actual_vs_pred": url_for("run_file", run_id=run_id, filename="actual_vs_predicted.png"),
        "residual": url_for("run_file", run_id=run_id, filename="residual_distribution.png"),
    }

    download_urls = {
        "metrics_csv": url_for("run_file", run_id=run_id, filename="model_metrics.csv"),
        "predictions_csv": url_for("run_file", run_id=run_id, filename="best_model_predictions.csv"),
    }

    return render_template(
        "index.html",
        success=True,
        run_id=run_id,
        best_model=result["best_model_name"],
        best_metrics=result["best_metrics"],
        race_time_seconds=result["race_time_prediction"],
        race_time_hms=result["prediction_hms"],
        predict_distance=predict_distance,
        metrics_table=result["results"].round(4).to_dict(orient="records"),
        chart_urls=chart_urls,
        download_urls=download_urls,
    )


@app.get("/runs/<run_id>/<path:filename>")
def run_file(run_id: str, filename: str):
    run_dir = OUTPUT_ROOT / run_id
    return send_from_directory(run_dir, filename, as_attachment=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
