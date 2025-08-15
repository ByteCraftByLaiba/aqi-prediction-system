# 🌏 AQI Prediction System

An end-to-end **Air Quality Index (AQI) forecasting pipeline** that:
- Collects and updates air pollution datasets using a **sliding window** of recent history.
- Trains **optimized ML models** (CatBoost, LightGBM, XGBoost) for PM2.5 and PM10 forecasts at multiple horizons (3h, 6h).
- Generates **feature importance (SHAP) summaries** for model explainability.
- Saves all outputs (models, datasets, metrics) in structured directories.
- Runs **automatically every day** via GitHub Actions for continuous model updates.

---

## 📂 Project Structure

```

.
├── datasets\_per\_target/      # CSV datasets (one per target variable)
├── models/                   # Saved model files (.joblib) + metrics.json
│   ├── shap/                 # SHAP summary CSVs per target
├── scripts/
│   ├── data\_collect\_update.py   # Collect & update latest data (sliding window)
│   ├── train\_daily.py           # Train all targets with AutoML pipeline
│   ├── main\_single.py           # Train one target at a time
│   ├── main\_all.py              # Train all targets sequentially
├── model\_pipeline.py         # AutoMLPipeline implementation
├── requirements.txt
└── .github/
└── workflows/
└── daily\_pipeline.yml   # GitHub Actions daily retrain workflow

````

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<YOUR_USER>/<YOUR_REPO>.git
cd <YOUR_REPO>
````

### 2️⃣ Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Mac/Linux
.venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Usage

### 🔹 Update Datasets (Sliding Window)

```bash
python scripts/data_collect_update.py
```

> This pulls the latest available data and replaces the oldest records to keep a fixed time window (default: last 24 months).

### 🔹 Train All Models (Manual Run)

```bash
python scripts/train_daily.py
```

This will:

* Train **pm2\_5\_t+3h**, **pm2\_5\_t+6h**, **pm10\_t+3h**, **pm10\_t+6h**.
* Save:

  * Best model `.joblib` per target in `models/`
  * SHAP summaries in `models/shap/`
  * Updated `metrics.json` with performance stats

### 🔹 Train a Single Target

```bash
# Example: Train pm2_5_t+3h
TARGET="pm2_5_t+3h" python scripts/main_single.py
```

---

## 📊 Output Files

### `models/metrics.json`

Example entry:

```json
{
  "pm2_5_t+3h": {
    "model": "catboost_regressor",
    "cv_score": 0.9422,
    "rmse": 11.50,
    "mse": 132.38,
    "r2": 0.9422,
    "mae": 7.97,
    "trained_at": "2025-08-15T07:28:48.632058Z",
    "model_path": "models/pm2_5_tplus3h_best.joblib",
    "shap_summary": "models/shap/pm2_5_tplus3h_shap.csv"
  }
}
```

---

## 🤖 Automation (CI/CD)

The **GitHub Actions workflow** in `.github/workflows/daily_pipeline.yml` runs daily at `03:00 UTC` (08:00 PKT) and will:

1. Checkout the repository.
2. Install Python & dependencies.
3. Run `data_collect_update.py` to fetch fresh data.
4. Run `train_daily.py` to train all models.
5. Upload updated datasets, models, and metrics as workflow artifacts.

### Manual Trigger

You can manually trigger a run from the **Actions** tab → *Daily AQI Data & Retrain* → *Run workflow*.

---

## 📡 API Integration

Once trained, models in `models/*.joblib` can be used for predictions via your FastAPI service:

```python
import joblib
import pandas as pd

model = joblib.load("models/pm2_5_tplus3h_best.joblib")
X = pd.DataFrame([...])  # your feature row
prediction = model.predict(X)
```

---

## 🔒 Secrets & Environment Variables

If your data collection script requires API keys:

* Add them in **Settings → Secrets and variables → Actions** as `MY_API_KEY`.
* Reference them in the workflow step:

```yaml
env:
  API_KEY: ${{ secrets.MY_API_KEY }}
```

---

## 🛠 Maintenance Tips

* **Change retrain schedule**: Edit `cron` in `.github/workflows/daily_pipeline.yml`.
* **Extend sliding window**: Update `MONTHS_BACK` in the data collection script.
* **Add new targets**: Drop CSV in `datasets_per_target/` and update `BEST_MODEL_PER_TARGET`.

---

## 📌 Roadmap

* [ ] Deploy API to cloud for real-time AQI predictions
* [ ] Add ensemble models for higher accuracy
* [ ] Include weather forecast features
* [ ] Interactive dashboard for AQI trends

---

## 📜 License

MIT License © 2025 \Laiba Shahab

```
