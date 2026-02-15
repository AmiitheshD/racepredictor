# Race Predictor Lab (Web)

A Flask web app that lets you upload a running CSV, train multiple prediction models, and view/download charts and model metrics.

## Features
- CSV upload and one-click model training
- Model comparison (Linear, Ridge, GradientBoosting, RandomForest, Power-law baseline)
- Cross-validation metrics (RMSE, MAE, R2, MAPE)
- Race forecast for target distance (default 26.2)
- Charts generated automatically:
  - RMSE comparison
  - Actual vs Predicted
  - Residual distribution

## Local run

```bash
cd "/Users/amiitheshdhanuskodi/Documents/race predictor"
python3 -m pip install -r requirements.txt
python3 app.py
```

Open `http://127.0.0.1:8000`.

## Deploy (Render)

1. Create a GitHub repo and push this folder.
2. In Render, click **New +** -> **Web Service**.
3. Connect the repo.
4. Render will detect `render.yaml` and use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
5. Deploy.

## CLI model run

```bash
MPLCONFIGDIR=/tmp/matplotlib python3 race_predictor.py \
  --csv "/path/to/data.csv" \
  --target "Time" \
  --distance-col "Distance" \
  --predict-distance 26.2
```
