from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

PREDICTION_COUNT = Counter(
    "model_prediction_total",
    "Total number of predictions",
    ["model_version", "prediction_class"],
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Latency of predictions in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Total number of prediction errors",
)

MODEL_VERSION_GAUGE = Gauge(
    "model_version_info",
    "Current model version deployed",
)
