from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .data_utils import REQUIRED_FEATURES, TARGET_COLUMN, default_processed_path, project_root

MODEL_PATH = project_root() / "models" / "soil_fertility_model.joblib"
MODEL_METADATA_PATH = project_root() / "models" / "model_metadata.json"
MODEL_NAME = "Random Forest Classifier"
CLASS_ORDER = ["Low", "Medium", "High"]
RANDOM_STATE = 42


def prepare_ml_data(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Validate and select the exact numeric feature contract used by training and inference."""
    required_columns = [*REQUIRED_FEATURES, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required model columns: {', '.join(missing_columns)}")
    if dataframe.empty:
        raise ValueError("The processed dataset is empty.")

    features = dataframe[REQUIRED_FEATURES].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        invalid_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Model features contain missing or non-numeric values: {', '.join(invalid_columns)}")

    target = dataframe[TARGET_COLUMN].astype(str).str.strip()
    invalid_labels = sorted(set(target) - set(CLASS_ORDER))
    if target.isna().any() or invalid_labels:
        detail = f" Invalid labels: {', '.join(invalid_labels)}." if invalid_labels else ""
        raise ValueError(f"Fertility must contain only Low, Medium, or High.{detail}")
    if target.nunique() < 2:
        raise ValueError("At least two fertility classes are required for classification.")
    return features, target, REQUIRED_FEATURES.copy()


def split_ml_data(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    features, target, feature_columns = prepare_ml_data(dataframe)
    class_counts = target.value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each fertility class needs at least two samples for a stratified split.")
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )
    except ValueError as error:
        raise ValueError(f"The dataset is too small for a stratified train/test split: {error}") from error
    return x_train, x_test, y_train, y_test, feature_columns


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    if len(x_train) < 2 or y_train.nunique() < 2:
        raise ValueError("Training requires at least two samples and two classes.")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=CLASS_ORDER).tolist()
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, labels=CLASS_ORDER, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, labels=CLASS_ORDER, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, labels=CLASS_ORDER, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": matrix,
        "class_order": CLASS_ORDER,
        "test_class_distribution": {str(label): int(count) for label, count in y_test.value_counts().items()},
        "feature_importance": {
            column: float(importance)
            for column, importance in zip(feature_columns or list(x_test.columns), model.feature_importances_)
        },
    }


def save_model(model: RandomForestClassifier, path: str | Path = MODEL_PATH) -> Path:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_model(path: str | Path = MODEL_PATH) -> RandomForestClassifier:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model was not found: {model_path}")
    model = joblib.load(model_path)
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise ValueError("The stored file is not a compatible classification model.")
    return model


def predict_fertility(model: RandomForestClassifier, input_data: pd.DataFrame | dict[str, float]) -> dict[str, Any]:
    if isinstance(input_data, dict):
        input_frame = pd.DataFrame([input_data])
    else:
        input_frame = input_data.copy()
    missing_columns = [column for column in REQUIRED_FEATURES if column not in input_frame.columns]
    if missing_columns:
        raise ValueError(f"Prediction input is missing: {', '.join(missing_columns)}")
    ordered_input = input_frame[REQUIRED_FEATURES].apply(pd.to_numeric, errors="coerce")
    if ordered_input.isna().any().any():
        raise ValueError("Prediction inputs must be numeric and cannot be empty.")
    predicted_class = str(model.predict(ordered_input)[0])
    probabilities = model.predict_proba(ordered_input)[0]
    model_classes = [str(label) for label in model.classes_]
    probability_map = {label: 0.0 for label in CLASS_ORDER}
    probability_map.update({label: float(probability) for label, probability in zip(model_classes, probabilities)})
    return {
        "prediction": predicted_class,
        "confidence": probability_map[predicted_class],
        "probabilities": probability_map,
    }


def train_and_save_model(
    dataframe: pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = MODEL_METADATA_PATH,
    random_state: int = RANDOM_STATE,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    x_train, x_test, y_train, y_test, feature_columns = split_ml_data(dataframe, random_state=random_state)
    model = train_model(x_train, y_train, random_state=random_state)
    metrics = evaluate_model(model, x_test, y_test, feature_columns)
    save_model(model, model_path)
    metadata = {
        "model_name": MODEL_NAME,
        "features": feature_columns,
        "target": TARGET_COLUMN,
        "classes": CLASS_ORDER,
        "training_samples": int(len(x_train)),
        "testing_samples": int(len(x_test)),
        "training_class_distribution": {str(label): int(count) for label, count in y_train.value_counts().items()},
        "random_state": random_state,
        "prototype_dataset": True,
        **metrics,
    }
    metadata_file = Path(metadata_path)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model, metadata


def train_from_processed_dataset(
    processed_path: str | Path = default_processed_path(),
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = MODEL_METADATA_PATH,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    dataset_path = Path(processed_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset was not found: {dataset_path}")
    dataframe = pd.read_csv(dataset_path)
    return train_and_save_model(dataframe, model_path, metadata_path)
