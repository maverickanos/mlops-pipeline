# MLOps Production Pipeline

![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Monitoring-FF4B4B?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn)
![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?logo=prometheus)

End-to-end MLOps production pipeline with experiment tracking, model registry, REST API serving, monitoring dashboard, and containerized deployment.

---

## MLOps Lifecycle

```
Data Ingestion → Preprocessing → Experiment Tracking (MLflow) → Model Registry → API Serving → Monitoring & Drift Detection
       ↑                                                                                                     │
       └───────────────────────────────────── Retrain ───────────────────────────────────────────────────────┘
```

## Features

- **Experiment Tracking** — MLflow auto-logging with metrics (accuracy, precision, recall, F1, ROC-AUC)
- **Model Registry** — Versioned model storage with stage transitions (Staging → Production)
- **Hyperparameter Tuning** — GridSearchCV with configurable param grids
- **REST API Serving** — FastAPI with `/predict`, `/predict-batch`, `/model-info`, `/health`, `/metrics`
- **Monitoring Dashboard** — Streamlit UI with performance trends, drift alerts, retrain trigger
- **Data Drift Detection** — KS test & Population Stability Index (PSI) for feature distribution shifts
- **Prometheus Metrics** — Prediction count, latency, error counters with Grafana dashboards
- **Docker Deployment** — Multi-service orchestration (MLflow, API, Monitoring, Prometheus, Grafana)
- **CI/CD Ready** — Test suite included, containerized, version-controlled

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Training   │────▶│   MLflow     │────▶│  FastAPI     │────▶│  Prometheus  │
│  Container   │     │   Server     │     │  Serving     │     │  & Grafana   │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                           │                      │
                           ▼                      ▼
                    ┌──────────────┐     ┌──────────────┐
                    │   Model      │     │  Streamlit   │
                    │   Registry   │     │  Dashboard   │
                    └──────────────┘     └──────────────┘
```

## Quick Start (Docker Compose)

```bash
# Clone the repository
git clone https://github.com/roohan-514/mlops-production-pipeline.git
cd mlops-production-pipeline

# Start all services
docker-compose up -d

# Access services:
# - MLflow UI:     http://localhost:5000
# - API:           http://localhost:8000
# - Dashboard:     http://localhost:8501
# - Prometheus:    http://localhost:9090
# - Grafana:       http://localhost:3000 (admin/admin)
```

## Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Train a model (in another terminal)
python -m pipeline.train

# Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start monitoring dashboard
streamlit run monitoring/dashboard.py --server.port 8501
```

## Training a New Model

```bash
python -m pipeline.train
```

Configure training parameters in `pipeline/config.py`. The script:
1. Loads and preprocesses the dataset (Breast Cancer by default)
2. Runs GridSearchCV with Random Forest
3. Logs parameters, metrics, and model artifacts to MLflow
4. Registers the best model in the MLflow Model Registry

## Running the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

| Method | Endpoint            | Description                    |
|--------|---------------------|--------------------------------|
| POST   | `/predict`          | Single prediction              |
| POST   | `/predict-batch`    | Batch predictions              |
| GET    | `/model-info`       | Model metadata                 |
| GET    | `/health`           | Health check                   |
| GET    | `/metrics`          | Prometheus metrics             |

### Example API Calls

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]}'

# Batch prediction
curl -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1.0, ...], [2.0, ...]]}'

# Health check
curl "http://localhost:8000/health"
```

### Interactive API Docs

Open `http://localhost:8000/docs` in your browser for Swagger UI.

## Monitoring Dashboard

The Streamlit dashboard provides:
- **Overview** — Model version, run count, API status
- **Model Performance** — Metrics across MLflow runs
- **Prediction Distribution** — Class distribution and confidence scores
- **Data Drift Detection** — KS test and PSI analysis per feature
- **Recent Predictions** — Prediction log viewer
- **Retrain Model** — One-click retraining trigger

## Project Structure

```
mlops-production-pipeline/
├── pipeline/             # Training pipeline
│   ├── train.py          # Model training with MLflow
│   ├── evaluate.py       # Model evaluation & comparison
│   ├── data_preprocessing.py  # Data loading & preprocessing
│   └── config.py         # Training configuration
├── api/                  # REST API serving
│   ├── main.py           # FastAPI application
│   ├── config.py         # API configuration
│   └── schemas.py        # Pydantic models
├── monitoring/           # Monitoring & observability
│   ├── dashboard.py      # Streamlit dashboard
│   ├── metrics.py        # Prometheus metrics
│   └── drift_detection.py  # Drift detection
├── data/                 # Datasets
│   ├── sample_data.py    # Data generation
│   └── README.md
├── tests/                # Test suite
│   ├── test_pipeline.py  # Pipeline tests
│   └── test_api.py       # API tests
├── notebooks/            # Jupyter notebooks
├── Dockerfile.api        # API container
├── Dockerfile.train      # Training container
├── docker-compose.yml    # Multi-service orchestration
├── prometheus.yml        # Prometheus config
├── requirements.txt      # Python dependencies
└── README.md
```

## Tech Stack

| Component       | Technology                  |
|----------------|-----------------------------|
| Experiment Tracking | MLflow                  |
| Model Training  | Scikit-learn, Pandas        |
| API Serving     | FastAPI, Uvicorn            |
| Monitoring UI   | Streamlit                   |
| Observability   | Prometheus, Grafana         |
| Deployment      | Docker, Docker Compose      |
| Drift Detection | SciPy (KS test, PSI)        |
| Data Processing | Scikit-learn, Pandas, NumPy |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT
