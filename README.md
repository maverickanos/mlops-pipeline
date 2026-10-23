
# MLOps Production Pipeline

An end-to-end machine learning operations project for training, tracking, serving, and monitoring machine-learning models. The pipeline combines experiment tracking, model versioning, REST API serving, drift detection, service metrics, and containerized deployment.

---

## Overview

This project demonstrates a practical machine learning lifecycle:

1. Load and preprocess data.
2. Train and tune a model.
3. Track experiments and metrics with MLflow.
4. Register the selected model.
5. Serve predictions through FastAPI.
6. Monitor service metrics and feature drift.
7. Run the application locally using Docker Compose.

The default training configuration uses the Breast Cancer Wisconsin dataset and a Random Forest classifier. Training settings are defined in `pipeline/config.py`.

## Architecture

```text
                ┌──────────────────────┐
                │     Input Dataset    │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │ Preprocessing &      │
                │ Model Training       │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │ MLflow Experiments   │
                │ & Model Registry     │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │ FastAPI Prediction   │
                │ Service              │
                └──────────┬───────────┘
                           ▼
                ┌──────────────────────┐
                │ Metrics & Monitoring │
                └──────────────────────┘
```

## Features

- **Model Training:** Scikit-learn pipeline with configurable hyperparameter tuning using `GridSearchCV`.
- **Experiment Tracking:** MLflow logging for parameters, metrics, and model artifacts.
- **Model Registry:** Versioned model storage and model-stage management.
- **REST API:** FastAPI endpoints for single predictions, batch predictions, model metadata, health checks, and Prometheus metrics.
- **Monitoring Dashboard:** Streamlit interface for viewing experiment results, prediction distributions, and drift checks.
- **Data Drift Detection:** Kolmogorov–Smirnov (KS) tests and Population Stability Index (PSI) for detecting changes in feature distributions.
- **Observability:** Prometheus metrics for request counts, latency, and errors, with Grafana dashboard support.
- **Containerized Deployment:** Docker Compose configuration for running the application and supporting services.
- **Automated Testing:** Test suite for model pipeline and API behavior.

> **Note:** Drift alerts indicate potential changes in input data; they do not automatically prove that model quality has degraded. Validate retraining and model-promotion workflows before using them beyond a local demonstration.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Model Training | Scikit-learn, Pandas, NumPy |
| Experiment Tracking | MLflow |
| API Serving | FastAPI, Uvicorn, Pydantic |
| Monitoring UI | Streamlit |
| Metrics | Prometheus, Grafana |
| Drift Detection | SciPy, KS Test, PSI |
| Deployment | Docker, Docker Compose |
| Testing | pytest |

## Getting Started

### Prerequisites

- Git
- Python 3.11
- Docker Engine or Docker Desktop with Docker Compose
- Local ports `5000`, `8000`, `8501`, `9090`, and `3000` available

### Clone the Repository

```bash
git clone https://github.com/roohan-514/mlops-production-pipeline.git
cd mlops-production-pipeline
```

### Option 1: Run with Docker Compose

```bash
docker compose up --build
```

When the services are running, access them at:

| Service | URL |
|---|---|
| MLflow | http://localhost:5000 |
| FastAPI | http://localhost:8000 |
| Interactive API Docs | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Use the Grafana credentials configured in your local environment. Do not expose default credentials or monitoring interfaces publicly.

Stop the services with:

```bash
docker compose down
```

### Option 2: Run Locally

Create and activate a virtual environment:

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the MLflow server:

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db
```

In another terminal, train the model:

```bash
python -m pipeline.train
```

Start the FastAPI application:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Start the monitoring dashboard:

```bash
streamlit run monitoring/dashboard.py --server.port 8501
```

Make sure the required services are running before opening the dashboard or calling the API.

## Model Training & Experiment Tracking

Run the training pipeline:

```bash
python -m pipeline.train
```

The workflow is designed to:

1. Load and preprocess the configured dataset.
2. Train and tune a Random Forest classifier using `GridSearchCV`.
3. Log parameters, metrics, and artifacts to MLflow.
4. Register the selected model in the MLflow Model Registry.

Use the MLflow UI to inspect runs and compare experiments.

Training settings can be modified in `pipeline/config.py`.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Generate a prediction for one input |
| `POST` | `/predict-batch` | Generate predictions for multiple inputs |
| `GET` | `/model-info` | Return model metadata |
| `GET` | `/health` | Check API health |
| `GET` | `/metrics` | Expose Prometheus metrics |

Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation and the authoritative request schemas.

### Example Requests

The exact request format depends on `api/schemas.py` and the configured dataset. If `/predict` accepts a `features` array, a request may look like this:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

Replace the example values with the correct number and order of features for the active model. Use Swagger UI to confirm the expected schema.

Check API health:

```bash
curl http://localhost:8000/health
```

## Monitoring & Drift Detection

The Streamlit dashboard is intended to provide views for:

- Model versions and experiment runs
- Training metrics
- Prediction distributions and confidence scores, where available
- Recent prediction records
- Feature-level drift checks using KS tests and PSI
- A retraining control, if enabled in the current configuration

Drift checks compare incoming feature distributions with a reference distribution. Investigate alerts and evaluate labeled outcomes where possible before deciding whether a model requires retraining.

## Repository Structure

```text
mlops-production-pipeline/
├── pipeline/
│   ├── train.py                 # Model training and MLflow logging
│   ├── evaluate.py              # Model evaluation and comparison
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   └── config.py                # Training configuration
├── api/
│   ├── main.py                  # FastAPI application
│   ├── config.py                # API configuration
│   └── schemas.py               # Request/response schemas
├── monitoring/
│   ├── dashboard.py             # Streamlit dashboard
│   ├── metrics.py               # Prometheus metrics
│   └── drift_detection.py       # KS test and PSI checks
├── data/
│   ├── sample_data.py           # Sample data utilities
│   └── README.md
├── tests/
│   ├── test_pipeline.py         # Pipeline tests
│   └── test_api.py              # API tests
├── notebooks/                   # Exploration notebooks
├── Dockerfile.api
├── Dockerfile.train
├── docker-compose.yml
├── prometheus.yml
├── requirements.txt
└── README.md
```

## Testing

Run the test suite from the repository root:

```bash
pytest
```

Add tests for valid inputs, invalid inputs, missing values, model-loading failures, and prediction behavior as the project evolves.

## CI/CD

A typical GitHub Actions workflow should:

1. Install project dependencies.
2. Run linting and unit tests.
3. Validate the training and API code.
4. Build Docker images.
5. Optionally publish images or deploy after all checks pass.

Only describe CI/CD as enabled once the relevant workflow files are present and passing in the repository.

## Contributing

1. Fork the repository.
2. Create a focused branch:

   ```bash
   git checkout -b feature/your-change
   ```

3. Implement your change and add or update tests.
4. Run the test suite.
5. Commit and push your branch.
6. Open a pull request describing the change and validation performed.

## License

This project declares the MIT License. Refer to the repository's `LICENSE` file for the complete terms.

---

If you use this project as a portfolio foundation, document the code you implemented, the experiments you ran, and the results you measured. Avoid claiming performance improvements or production readiness that have not been verified.
