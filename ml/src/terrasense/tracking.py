import mlflow

from terrasense.config import PROJECT_ROOT

EXPERIMENT_NAME = "terrasense"


def setup_mlflow() -> None:
    """Point MLflow at a local SQLite store so runs persist without a tracking server."""
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        # artifacts go next to the db instead of ./mlruns of whatever cwd we run from
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=(PROJECT_ROOT / "mlruns").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
