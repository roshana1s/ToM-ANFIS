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

---

## API: Endpoints, Observation Window & Prediction Horizon

### Quick Concepts

**Observation Window** (default: 15 frames)
- Consecutive frames aggregated to compute temporal ToM features (B3, B4, G2, A1, A2, C1, C2)
- Each feature value is the mean/ratio over these 15 frames
- Training notebook & current model use `OBSERVATION_WINDOW=15`

**Prediction Horizon** (default: 30 frames)
- How many frames into the future the model predicts for
- `prediction_horizon=30` means: given 15 frames of observation, predict the crossing state 30 frames ahead
- Total frames needed: 15 + 30 = 45 frames minimum

**Frame Timeline Visualization**
```
[Observation: 15 frames] [Future Horizon: 30 frames] [Crossing Event]
Frames 0-14             Frames 15-44                Frame 45 (or 59 in longer video)

Model observes frames 0-14, predicts: "30 frames from now, will pedestrian cross?"
```

**Real-world Example: 60-frame video**
```
Video: frame_0 ... frame_29 ... frame_59 (crossing occurs)

With obs=15, horizon=30:
  Prediction frame = 59 - 30 = 29
  Observation window = frames [15..29] (last 15 frames before prediction)
  Model predicts: "At frame 29, will crossing occur 30 frames from now?"
  Answer: Yes, at frame 59 (which is 45 frames from window start)
```

**Frame Requirement**
- API requires **minimum 45 frames** (obs_window + prediction_horizon) to avoid truncation
- Fewer frames will fail validation with clear error message

---

### API Endpoints

#### 1. POST /predict/from-annotations (Raw Annotations)

**Purpose:** Accept raw PIE-style annotations, extract ToM features, then run inference.

**Required Fields:**
- `age` (int 0-3): pedestrian age category
- `intersection` (int 0-4): intersection type
- `frames` (list of ints): frame IDs in order
- `bbox` (list of [x1,y1,x2,y2]): pedestrian bounding boxes per frame
- `occlusion` (list of ints): occlusion level each frame (0=visible, 1=partial, 2=hidden)
- `behavior` (object): action, gesture, look lists (one value per frame)
- `traffic_objects` (list): vehicles, crosswalks, traffic lights with obj_class, frames, optional bbox/state
- `obd_speed` (dict): mapping frame_id → speed (m/s)

**Optional Fields:**
- `observation_window` (int): must equal configured default (15)
- `prediction_horizon` (int): override default (30) for different future time

**Example Payload** (minimum valid request):
```json
{
	"age": 2,
	"intersection": 0,
	"frames": [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
	"bbox": [[100,200,150,260], [101,200,151,260], [102,200,152,260], [103,200,153,260], [104,200,154,260], [105,200,155,260], [106,200,156,260], [107,200,157,260], [108,200,158,260], [109,200,159,260], [110,200,160,260], [111,200,161,260], [112,200,162,260], [113,200,163,260], [114,200,164,260]],
	"occlusion": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	"behavior": {
		"action": [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
		"gesture": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
		"look": [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1]
	},
	"traffic_objects": [],
	"obd_speed": {"15": 5.2, "16": 5.1, "17": 5.0, "18": 4.9, "19": 4.8, "20": 4.7, "21": 4.6, "22": 4.5, "23": 4.4, "24": 4.3, "25": 4.2, "26": 4.1, "27": 4.0, "28": 3.9, "29": 3.8},
	"observation_window": 15,
	"prediction_horizon": 30
}
```

**Response:**
```json
{
	"predicted_crossing_probability": 0.87,
	"intention_level": "HIGH",
	"top_fired_rules": [
		{
			"rule_id": 1234,
			"firing_strength": 0.456,
			"rule": "IF A1 is High AND A2 is High THEN crossing intention is HIGH",
			"consequent_linguistic": "HIGH"
		}
	]
}
```

---

#### 2. POST /predict/from-tom-features (Pre-extracted Features)

**Purpose:** Accept pre-computed 9-dimensional ToM feature vector and run ANFIS inference directly. Use this endpoint when features are already extracted or computed externally.

**Required Fields:**
- `B1_age_vulnerability_awareness` (float [0, 1]): age-group vulnerability factor
- `B2_intersection_complexity` (float [0, 1]): intersection layout complexity
- `B3_pedestrian_position_proximity` (float [0, 1]): proximity to vehicle/curb
- `B4_vehicle_density` (float [0, 1]): density of surrounding vehicles
- `G2_urgency_rushing` (float [0, 1]): behavioral rushing indicator
- `A1_vehicle_awareness` (float [0, 1]): attention to vehicles
- `A2_traffic_signal_awareness` (float [-1, 1]): attention to traffic signals (-1=looking away, +1=looking towards)
- `C1_relative_vehicle_speed` (float [0, 1]): normalized vehicle speed
- `C2_occlusion_factor` (float [0, 1]): visibility/occlusion level

**Example Payload:**
```json
{
	"B1_age_vulnerability_awareness": 0.3,
	"B2_intersection_complexity": 0.5,
	"B3_pedestrian_position_proximity": 0.8,
	"B4_vehicle_density": 0.1,
	"G2_urgency_rushing": 0.4,
	"A1_vehicle_awareness": 0.9,
	"A2_traffic_signal_awareness": 1.0,
	"C1_relative_vehicle_speed": 0.2,
	"C2_occlusion_factor": 0.0
}
```

**Response:** Same as Endpoint 1 (crossing probability, intention level, top fired rules)

---

## Testing & Validation

### Interactive API Testing
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

Both provide live request/response testing for both endpoints.

### Frame Count Validation
- **Minimum frames required:** 45 (observation_window + prediction_horizon = 15 + 30)
- **Per-frame list alignment:** All lists (`bbox`, `occlusion`, `behavior.*`, `obd_speed` keys) must match `len(frames)`
- **Frame order:** Frames must be in ascending sequential order

### Feature Range Validation (Endpoint 2)
- **Normal range [0, 1]:** B1, B2, B3, B4, G2, A1, C1, C2
- **Extended range [-1, 1]:** A2 (signal awareness can be negative)
- Submit invalid ranges → API returns validation error with field guidance

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Insufficient frames` | `len(frames) < 45` | Increase frames to ≥ 45 |
| `List length mismatch` | `len(bbox) ≠ len(frames)` | Ensure all per-frame lists match frame count |
| `Value out of range` | Feature outside valid bounds | Clamp feature to [0,1] or [-1,1] for A2 |
| `Invalid frame order` | Frames not ascending | Sort frames before submission |

---

## Model Architecture & Inference

The ANFIS model uses a 5-layer neuro-fuzzy architecture:

1. **Input Layer:** 9 normalized ToM features
2. **Membership Functions:** Gaussian curves for feature fuzzification
3. **Rule Layer:** 19,683 fuzzy IF-THEN rules (3⁹ combinations)
4. **Normalization & Inference:** Weighted rule firing
5. **Output Layer:** Sigmoid-activated crossing probability ∈ [0, 1]

The `predict_explain()` method returns:
- **predicted_crossing_probability** (float): sigmoid output indicating likelihood of crossing
- **intention_level** (str): "LOW", "MEDIUM", or "HIGH" based on threshold bands
- **top_fired_rules** (list): 3 most relevant rules with firing strengths and linguistic descriptions

---

## References

- **Training Notebook:** `notebooks/ToM_ANFIS_training_pipeline.ipynb`
- **Core Model:** `app/core/anfis.py`
- **Feature Extractor:** `app/core/extractor.py`
- **PIE Dataset:** https://github.com/aras62/PIE

---

## Notes

- **Off-by-one behavior:** The current extractor uses `prediction_horizon=30` to observe features at frame position 29 when a crossing event occurs at frame 59 (i.e., 30 frames in the future). This is by design to avoid boundary truncation with the trained model. See "Frame Timeline Visualization" above for examples.
- **Model serialization:** The trained ANFIS model and MinMaxScaler are loaded from disk at startup (`app/artifacts/`). Ensure both files exist before running the API.
- **Feature normalization:** All 9 features are normalized to [0, 1] using the fitted MinMaxScaler before inference.