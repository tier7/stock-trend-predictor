import json
import re
from pathlib import Path
from uuid import uuid4

import joblib

from services.ml.dataset_service import FEATURE_COLUMNS


GLOBAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "global_forest"
MODEL_FORMAT_VERSION = 1
MODEL_ID_PATTERN = re.compile(r"global_forest_\d{4}-\d{2}-\d{2}_[0-9a-f]{8}")


def model_path(model_id, model_dir=None):
    if not isinstance(model_id, str) or not MODEL_ID_PATTERN.fullmatch(model_id):
        raise ValueError("Nieprawidłowy identyfikator modelu globalnego.")

    return Path(model_dir or GLOBAL_MODEL_DIR) / f"{model_id}.joblib"


def save_global_model(model, metadata, model_dir=None):
    metadata = dict(metadata)
    metadata["model_id"] = f"global_forest_{metadata['training_cutoff']}_{uuid4().hex[:8]}"
    metadata["format_version"] = MODEL_FORMAT_VERSION
    path = model_path(metadata["model_id"], model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(".json")
    temporary_model = path.with_suffix(".joblib.tmp")
    temporary_metadata = path.with_suffix(".json.tmp")

    try:
        joblib.dump({"model": model, "metadata": metadata}, temporary_model, compress=3)
        temporary_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        temporary_model.replace(path)
        temporary_metadata.replace(metadata_path)
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    return metadata


def list_global_models(model_dir=None):
    models = []
    directory = Path(model_dir or GLOBAL_MODEL_DIR)
    for path in directory.glob("global_forest_*.json"):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if (
                metadata["model_id"] == path.stem
                and model_path(path.stem, directory).is_file()
                and metadata["format_version"] == MODEL_FORMAT_VERSION
                and metadata["feature_columns"] == FEATURE_COLUMNS
            ):
                models.append(metadata)
        except (OSError, ValueError, KeyError, TypeError):
            continue

    return sorted(models, key=lambda item: item["created_at"], reverse=True)


def get_active_global_model(model_dir=None, model_id=None):
    models = list_global_models(model_dir)
    if model_id:
        for metadata in models:
            if metadata["model_id"] == model_id:
                return metadata
        raise FileNotFoundError("Nie znaleziono ustawionego modelu globalnego. Sprawdź GLOBAL_MODEL_ID.")
    if not models:
        raise FileNotFoundError("Brak gotowego modelu globalnego. Możesz uruchomić model lokalny.")
    return models[0]


def load_global_model(model_id, model_dir=None):
    path = model_path(model_id, model_dir)

    if not path.is_file():
        raise FileNotFoundError("Brak zapisanego modelu. Najpierw wytrenuj model globalny.")
    try:
        bundle = joblib.load(path)
        metadata = bundle["metadata"]
        model = bundle["model"]
        if (
            metadata["model_id"] != model_id
            or metadata["format_version"] != MODEL_FORMAT_VERSION
            or metadata["feature_columns"] != FEATURE_COLUMNS
            or list(model.classes_) != [0, 1]
        ):
            raise ValueError("Incompatible model")
    except Exception as error:
        raise ValueError("Nie można wczytać modelu. Wytrenuj nową wersję.") from error
    return model, metadata
