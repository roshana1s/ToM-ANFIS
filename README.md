# ToM-ANFIS
A Theory of Mind-Based Cognitive Framework with Adaptive Neuro-Fuzzy Inference for Explainable Pedestrian Crossing Intention Prediction

---

## File Structure

```
ToM-ANFIS/
├── app/
│   ├── artifacts/                  # Saved model weights and scaler
│   │   ├── anfis_model_optimized.pth
│   │   └── scaler_optimized.save
│   ├── core/                       # ANFIS model class and ToM feature extractor class
│   │   ├── anfis.py
│   │   └── extractor.py
│   ├── routes/                     # FastAPI route handlers
│   │   └── predict.py
│   ├── schemas/                    # Pydantic input/output schemas
│   │   ├── RawAnnotationInput.py
│   │   ├── ToMFeaturesInput.py
│   │   └── PredictionResponse.py
│   ├── services/                   # Business logic layer
│   │   ├── prediction_service.py
│   │   └── feature_extraction_service.py
│   ├── config.py                   # App settings (paths, thresholds)
│   └── main.py                     # FastAPI app entry point
├── notebooks/                      # Training notebooks
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

---

## Local Setup

### 1. Install uv

On Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

On macOS/Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and enter the project

```bash
git clone https://github.com/roshana1s/ToM-ANFIS.git
cd ToM-ANFIS
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Place model artifacts

Copy the saved model and scaler files into `app/artifacts/`:

```
app/artifacts/anfis_model_optimized.pth
app/artifacts/scaler_optimized.save
```

### 5. Run the API

```bash
uv run python -m app.main
```

The API will be available at `http://localhost:8000`.  
Interactive docs at `http://localhost:8000/docs`.

---

## Adding New Packages

```bash
uv add <package-name>
```

This updates `pyproject.toml` and `uv.lock` automatically.