import json
import re
from pathlib import Path
from uuid import uuid4
import torch
from sklearn.base import is_classifier, is_regressor
from services.ml.model_config import ModelConfig
import joblib
from services.ml.dataset_service import FEATURE_COLUMNS
from services.ml.feature_schema import get_feature_columns
from services.ml.news_feature_service import NEWS_FEATURE_POLICY
from services.ml.mlp_model import NeuralNetwork
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from copy import deepcopy
from services.ml.market_feature_service import MARKET_FEATURE_POLICY
from services.ml.sector_feature_service import (SECTOR_FEATURE_POLICY, SECTOR_BENCHMARKS)

GLOBAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "global_forest"
MODEL_FORMAT_VERSION = 1

TORCH_MODEL_DIR = (Path(__file__).resolve().parents[2] / "models" / "neural_networks")
TORCH_MODEL_FORMAT_VERSION = 1
LIGHTGBM_MODEL_DIR = (Path(__file__).resolve().parents[2] / "models" / "global_lightgbm")
MODEL_ID_PATTERN = re.compile(r"(global_forest|global_lightgbm)_\d{4}-\d{2}-\d{2}_[0-9a-f]{8}")


def model_path(model_id, model_dir=None):
    if not isinstance(model_id, str):
        raise ValueError("Invalid model ID")

    match = MODEL_ID_PATTERN.fullmatch(model_id)
    if match is None:
        raise ValueError("Invalid model ID")

    default_dir = (
        GLOBAL_MODEL_DIR
        if match.group(1) == "global_forest"
        else LIGHTGBM_MODEL_DIR
    )

    return Path(model_dir or default_dir) / f"{model_id}.joblib"


def save_global_model(model, metadata, model_dir=None):
    metadata = dict(metadata)
    config = get_model_config(metadata)

    if (
        config.algorithm not in ("random_forest", "lightgbm")
        or config.scope != "global"):
        raise ValueError("Expected a global tree model configuration")

    if "market" in config.feature_groups:
        metadata.setdefault("market_policy",deepcopy(MARKET_FEATURE_POLICY))
    if "sector" in config.feature_groups:
        metadata.setdefault("sector_policy", deepcopy(SECTOR_FEATURE_POLICY))
    validate_feature_policies(metadata, config)

    prefix = (
        "global_forest"
        if config.algorithm == "random_forest"
        else "global_lightgbm")

    metadata["model_id"] = (f"{prefix}_{metadata['training_cutoff']}_{uuid4().hex[:8]}")
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
            config = get_model_config(metadata)
            validate_feature_policies(metadata, config)
            expected_columns = get_feature_columns(config.feature_groups)
            if (
                metadata["model_id"] == path.stem
                and model_path(path.stem, directory).is_file()
                and metadata["format_version"] == MODEL_FORMAT_VERSION
                and metadata["feature_columns"] == expected_columns
            ):
                models.append(metadata)
        except (OSError, ValueError, KeyError, TypeError):
            continue

    return sorted(models, key=lambda item: item["created_at"], reverse=True)


def get_active_global_model(model_dir=None, model_id=None, *, task="classification", horizon=1):
    models = []
    for metadata in list_global_models(model_dir):
        config = get_model_config(metadata)

        if config.task == task and config.horizon == horizon:
            models.append(metadata)
    if model_id:
        for metadata in models:
            if metadata["model_id"] == model_id:
                return metadata
        raise FileNotFoundError("Nie znaleziono ustawionego modelu globalnego. Sprawdź GLOBAL_MODEL_ID.")
    if not models:
        raise FileNotFoundError("Brak gotowego modelu globalnego. Możesz uruchomić model lokalny.")
    return models[0]


def get_model_config(metadata):
    if "config" in metadata:
        return ModelConfig.from_dict(metadata["config"])

    return ModelConfig(
        algorithm="random_forest",
        task="classification",
        random_state=metadata.get("random_state", 26),
    )

def validate_feature_policies(metadata, config):
    if "news" in config.feature_groups:
        if metadata.get("news_policy") != NEWS_FEATURE_POLICY:
            raise ValueError("Incompatible news feature policy")

    if "market" in config.feature_groups:
        if metadata.get("market_policy") != MARKET_FEATURE_POLICY:
            raise ValueError("Incompatible market feature policy")

    if "sector" in config.feature_groups:
        if metadata.get("sector_policy") != SECTOR_FEATURE_POLICY:
            raise ValueError("Incompatible sector feature policy")

        benchmarks = metadata.get("sector_benchmarks")
        training_tickers = metadata.get("train_tickers")

        if not isinstance(benchmarks, dict) or not benchmarks:
            raise ValueError("Missing sector benchmark mapping")
        if (not isinstance(training_tickers, list) or not training_tickers or any(not isinstance(ticker, str) or not ticker.strip() for ticker in training_tickers)):
            raise ValueError("Invalid training ticker list")

        expected_tickers = {ticker.strip().upper() for ticker in training_tickers}
        if set(benchmarks.keys()) != expected_tickers:
            raise ValueError("Sector benchmark mapping must cover all training tickers")
        if any(benchmark not in SECTOR_BENCHMARKS.values() for benchmark in benchmarks.values()):
            raise ValueError("Unsupported sector benchmark")

def load_global_model(model_id, model_dir=None):
    path = model_path(model_id, model_dir)

    if not path.is_file():
        raise FileNotFoundError("Brak zapisanego modelu. Najpierw wytrenuj model globalny.")
    try:
        bundle = joblib.load(path)
        metadata = bundle["metadata"]
        model = bundle["model"]
        config = get_model_config(metadata)
        validate_feature_policies(metadata, config)
        expected_columns = get_feature_columns(config.feature_groups)
        if (
            metadata["model_id"] != model_id
            or metadata["format_version"] != MODEL_FORMAT_VERSION
            or metadata["feature_columns"] != expected_columns
        ):
            raise ValueError("Incompatible model")

        if list(model.feature_names_in_) != metadata["feature_columns"]:
            raise ValueError("Model features do not match metadata")



        if (config.algorithm not in ("random_forest", "lightgbm") or config.scope != "global"):
            raise ValueError("Incompatible model configuration")

        expected_prefix = ("global_forest" if config.algorithm == "random_forest" else "global_lightgbm")

        if not model_id.startswith(f"{expected_prefix}_"):
            raise ValueError("Model ID does not match configured algorithm")

        expected_types = (
            (RandomForestClassifier, RandomForestRegressor)
            if config.algorithm == "random_forest"
            else (LGBMClassifier, LGBMRegressor))

        if not isinstance(model, expected_types):
            raise ValueError("Model type does not match configured algorithm")

        if config.task == "classification":
            if not is_classifier(model) or list(model.classes_) != [0, 1]:
                raise ValueError("Expected a binary classifier")

        elif not is_regressor(model):
            raise ValueError("Expected a regressor")
    except Exception as error:
        raise ValueError("Nie można wczytać modelu. Wytrenuj nową wersję.") from error
    return model, metadata


def save_torch_model(model, scaler, metadata, model_dir=None):
    metadata = dict(metadata)
    config = ModelConfig.from_dict(metadata["config"])
    if "market" in config.feature_groups:
        metadata.setdefault("market_policy", deepcopy(MARKET_FEATURE_POLICY))
    if "sector" in config.feature_groups:
        metadata.setdefault("sector_policy", deepcopy(SECTOR_FEATURE_POLICY))
    metadata["format_version"] = TORCH_MODEL_FORMAT_VERSION
    metadata["model_id"] = (
        f"{metadata['architecture']}_"
        f"{metadata['training_cutoff']}_"
        f"{uuid4().hex[:8]}")

    validate_torch_metadata(metadata)

    directory = (Path(model_dir or TORCH_MODEL_DIR) / metadata["model_id"])
    directory.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), directory / "weights.pt")
    joblib.dump(scaler, directory / "scaler.joblib")
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metadata

def load_torch_model(model_id, model_dir=None):
    if not isinstance(model_id, str) or not re.fullmatch(r"mlp_\d{4}-\d{2}-\d{2}_[0-9a-f]{8}",model_id):
        raise ValueError("Incompatible model ID")

    directory = Path(model_dir or TORCH_MODEL_DIR) / model_id
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = validate_torch_metadata(metadata)


    if metadata["model_id"] != model_id:
        raise ValueError("Incompatible model ID")
    state_dict = torch.load(
        directory / "weights.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = NeuralNetwork(input_size=metadata["input_size"])
    model.load_state_dict(state_dict)
    scaler = joblib.load(directory / "scaler.joblib")

    if scaler.n_features_in_ != metadata["input_size"]:
        raise ValueError("Incompatible input size")
    if list(scaler.feature_names_in_) != metadata["feature_columns"]:
        raise ValueError("Incompatible feature columns")

    model.eval()
    return model, scaler, metadata

def validate_torch_metadata(metadata):

    if metadata.get("format_version") != TORCH_MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported MLP format. Train a new model.")

    config = ModelConfig.from_dict(metadata["config"])

    validate_feature_policies(metadata, config)

    if config.algorithm != "mlp":
        raise ValueError("Incompatible model configuration")

    if config.scope != "global":
        raise ValueError("Incompatible model configuration")

    if metadata["architecture"] != "mlp":
        raise ValueError("Incompatible model configuration")

    expected_columns = get_feature_columns(config.feature_groups)

    if metadata["feature_columns"] != expected_columns:
        raise ValueError("Model features do not match metadata")
    if metadata["input_size"] != len(expected_columns):
        raise ValueError("Model input sizes do not match metadata")

    if metadata["hidden_sizes"] != [64, 32]:
        raise ValueError("Unsupported MLP architecture")

    return config

