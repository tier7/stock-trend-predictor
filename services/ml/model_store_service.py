import json
import re
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
import torch
from sklearn.base import is_classifier, is_regressor
from services.ml.model_config import ModelConfig
import joblib
from services.ml.feature_schema import get_feature_columns
from services.ml.news_feature_service import NEWS_FEATURE_POLICY
from services.ml.mlp_model import NeuralNetwork
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from copy import deepcopy
from dataclasses import replace
from services.ml.market_feature_service import MARKET_FEATURE_POLICY
from services.ml.sector_feature_service import (SECTOR_FEATURE_POLICY, SECTOR_BENCHMARKS, extract_sector_benchmarks)
from services.ml.calibration_service import restore_calibrator

GLOBAL_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "global_forest"
MODEL_FORMAT_VERSION = 1
CALIBRATED_MODEL_FORMAT_VERSION = 2

TORCH_MODEL_DIR = (Path(__file__).resolve().parents[2] / "models" / "neural_networks")
TORCH_MODEL_FORMAT_VERSION = 1
LIGHTGBM_MODEL_DIR = (Path(__file__).resolve().parents[2] / "models" / "global_lightgbm")
HYBRID_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "hybrids"
HYBRID_MODEL_ID_PATTERN = re.compile(r"hybrid_(blend|stacking)_(\d{4}-\d{2}-\d{2})_[0-9a-f]{8}")
MODEL_ID_PATTERN = re.compile(r"(global_forest|global_lightgbm)_\d{4}-\d{2}-\d{2}_[0-9a-f]{8}")
TORCH_MODEL_ID_PATTERN = re.compile(r"mlp_\d{4}-\d{2}-\d{2}_[0-9a-f]{8}")


def build_training_metadata(final_train_df, feature_columns, config, training_cutoff, *, news_start=None, news_end=None):
    if final_train_df.empty:
        raise ValueError("Final training dataframe cannot be empty")
    metadata = {
        "config": config.to_dict(),
        "feature_columns": list(feature_columns),
        "training_cutoff": training_cutoff,
        "train_tickers": final_train_df["ticker"].drop_duplicates().tolist(),
        "training_rows": int(len(final_train_df)),
        "training_start_date": str(final_train_df["date"].min()),
        "training_end_date": str(final_train_df["date"].max()),
        "last_target_end_date": str(final_train_df["target_end_date"].max()),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    if config.task == "classification":
        metadata["baseline_prediction"] = int(final_train_df["target"].mode().iloc[0])
        metadata["target_definition"] = f"open(t+{config.horizon + 1}) > open(t+1)"
    else:
        metadata["baseline_return"] = float(final_train_df["target_return"].mean())
        metadata["target_definition"] = f"open(t+{config.horizon + 1}) / open(t+1) - 1"
    if "news" in config.feature_groups:
        metadata["news_policy"] = deepcopy(NEWS_FEATURE_POLICY)
        metadata["news_start"] = news_start
        metadata["news_end"] = news_end
    if "market" in config.feature_groups:
        metadata["market_policy"] = deepcopy(MARKET_FEATURE_POLICY)
    if "sector" in config.feature_groups:
        metadata["sector_policy"] = deepcopy(SECTOR_FEATURE_POLICY)
        metadata["sector_benchmarks"] = extract_sector_benchmarks(final_train_df)
    return metadata


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
    metadata = deepcopy(metadata)
    config = get_model_config(metadata)

    if (config.algorithm not in ("random_forest", "lightgbm") or config.scope != "global"):
        raise ValueError("Expected a global tree model configuration")

    if "market" in config.feature_groups:
        metadata.setdefault("market_policy",deepcopy(MARKET_FEATURE_POLICY))
    if "sector" in config.feature_groups:
        metadata.setdefault("sector_policy", deepcopy(SECTOR_FEATURE_POLICY))
    metadata["format_version"] = CALIBRATED_MODEL_FORMAT_VERSION if "calibration" in metadata else MODEL_FORMAT_VERSION
    validate_feature_policies(metadata, config)

    prefix = ("global_forest" if config.algorithm == "random_forest" else "global_lightgbm")

    metadata["model_id"] = (f"{prefix}_{metadata['training_cutoff']}_{uuid4().hex[:8]}")
    if "calibration" in metadata:
        metadata["calibration"]["model_id"] = metadata["model_id"]

    path = model_path(metadata["model_id"], model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(".json")
    temporary_model = path.with_suffix(".joblib.tmp")
    temporary_metadata = path.with_suffix(".json.tmp")

    try:
        joblib.dump({"model": model, "metadata": metadata}, temporary_model, compress=3)
        temporary_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
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
                and metadata["format_version"] in (MODEL_FORMAT_VERSION, CALIBRATED_MODEL_FORMAT_VERSION)
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
        random_state=metadata.get("random_state", 26))

def validate_feature_policies(metadata, config):
    if "calibration" in metadata:
        calibration = metadata["calibration"]
        restore_calibrator(calibration, config)
        if metadata.get("format_version") != CALIBRATED_MODEL_FORMAT_VERSION:
            raise ValueError("Calibration requires model format 2")
        if metadata.get("model_id") != calibration.get("model_id"):
            raise ValueError("Calibrator does not belong to this model")
        if metadata["training_cutoff"] > calibration["start_date"]:
            raise ValueError("Training overlaps calibration")
        if not set(calibration["tickers"]).issubset(metadata["train_tickers"]):
            raise ValueError("Calibration contains tickers outside the training cohort")
    elif metadata.get("format_version") == CALIBRATED_MODEL_FORMAT_VERSION:
        raise ValueError("Model format 2 requires calibration metadata")

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

def validate_model_id(model_id):
    if not isinstance(model_id, str) or not (MODEL_ID_PATTERN.fullmatch(model_id) or TORCH_MODEL_ID_PATTERN.fullmatch(model_id) or HYBRID_MODEL_ID_PATTERN.fullmatch(model_id)):
        raise ValueError("Invalid model ID")
    return model_id


def load_model(model_id, model_dir=None):
    validate_model_id(model_id)
    if model_id.startswith("hybrid_"):
        return load_hybrid_model(model_id, model_dir)
    if model_id.startswith("mlp_"):
        return load_torch_model(model_id, model_dir)
    model, metadata = load_global_model(model_id, model_dir)
    return model, None, metadata


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
            or metadata["format_version"] not in (MODEL_FORMAT_VERSION, CALIBRATED_MODEL_FORMAT_VERSION)
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
    metadata = deepcopy(metadata)
    config = ModelConfig.from_dict(metadata["config"])
    if "market" in config.feature_groups:
        metadata.setdefault("market_policy", deepcopy(MARKET_FEATURE_POLICY))
    if "sector" in config.feature_groups:
        metadata.setdefault("sector_policy", deepcopy(SECTOR_FEATURE_POLICY))
    metadata["format_version"] = CALIBRATED_MODEL_FORMAT_VERSION if "calibration" in metadata else TORCH_MODEL_FORMAT_VERSION
    metadata["model_id"] = (
        f"{metadata['architecture']}_"
        f"{metadata['training_cutoff']}_"
        f"{uuid4().hex[:8]}")
    if "calibration" in metadata:
        metadata["calibration"]["model_id"] = metadata["model_id"]

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
    if not isinstance(model_id, str) or not TORCH_MODEL_ID_PATTERN.fullmatch(model_id):
        raise ValueError("Incompatible model ID")

    directory = Path(model_dir or TORCH_MODEL_DIR) / model_id
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = validate_torch_metadata(metadata)


    if metadata["model_id"] != model_id:
        raise ValueError("Incompatible model ID")
    state_dict = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)

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

    if metadata.get("format_version") not in (TORCH_MODEL_FORMAT_VERSION, CALIBRATED_MODEL_FORMAT_VERSION):
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


def validate_hybrid_metadata(metadata, stacking_model=None):
    model_id = metadata.get("model_id")
    match = HYBRID_MODEL_ID_PATTERN.fullmatch(model_id) if isinstance(model_id, str) else None

    if match is None or metadata.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported hybrid model format or ID")
    method = metadata.get("evaluation_method")

    if method != match.group(1) or metadata.get("training_cutoff") != match.group(2):
        raise ValueError("Hybrid ID does not match its method or training cutoff")
    config = get_model_config(metadata)

    if config.scope != "global" or "calibration" in metadata or metadata.get("calibration_enabled", False):
        raise ValueError("Expected an uncalibrated global hybrid")
    validate_feature_policies(metadata, config)

    if metadata.get("feature_columns") != get_feature_columns(config.feature_groups):
        raise ValueError("Hybrid features do not match its configuration")
    algorithms = metadata.get("component_algorithms")

    if not isinstance(algorithms, list) or len(algorithms) < 2 or any(name not in ("random_forest", "mlp", "lightgbm") for name in algorithms):
        raise ValueError("Hybrid requires at least two supported component algorithms")
    if len(set(algorithms)) != len(algorithms):
        raise ValueError("Duplicate hybrid components")

    for field in ("component_model_ids", "component_training_details", "evaluation_parameters"):
        if not isinstance(metadata.get(field), dict) or set(metadata[field]) != set(algorithms):
            raise ValueError(f"Hybrid components do not match {field}")

    if any(not isinstance(metadata[field][name], dict) for field in ("component_training_details", "evaluation_parameters") for name in algorithms):
        raise ValueError("Invalid hybrid component details or parameters")

    prefixes = {"random_forest": "global_forest_", "mlp": "mlp_", "lightgbm": "global_lightgbm_"}

    for name in algorithms:
        component_id = validate_model_id(metadata["component_model_ids"][name])
        if not component_id.startswith(prefixes[name]):
            raise ValueError("Component ID does not match its algorithm")
    if method == "stacking":
        if stacking_model is None or list(getattr(stacking_model, "feature_names_in_", [])) != algorithms:
            raise ValueError("Stacking model features do not match the component order")
        if config.task == "classification":
            if not is_classifier(stacking_model) or list(stacking_model.classes_) != [0, 1]:
                raise ValueError("Expected a binary stacking classifier")
        elif not is_regressor(stacking_model):
            raise ValueError("Expected a stacking regressor")
    elif stacking_model is not None:
        raise ValueError("Blending does not use a stacking model")
    return config


def save_hybrid_model(result, metadata, model_dir=None):
    metadata = deepcopy(metadata)
    method = result["method"]
    algorithms = metadata.get("component_algorithms")
    components = result["model_results"]
    if method not in ("blend", "stacking") or metadata.get("evaluation_method") != method:
        raise ValueError("Incompatible hybrid method")
    if not isinstance(algorithms, list) or len(algorithms) < 2 or not isinstance(components, dict) or algorithms != list(components):
        raise ValueError("Hybrid results do not match the component order")
    config = get_model_config(metadata)
    if config.scope != "global" or "calibration" in metadata or metadata.get("calibration_enabled", False):
        raise ValueError("Expected an uncalibrated global hybrid")
    if any(component.get("calibrator") is not None for component in components.values()):
        raise ValueError("Calibrated components are not supported in this hybrid")
    metadata["model_id"] = f"hybrid_{method}_{metadata['training_cutoff']}_{uuid4().hex[:8]}"
    validate_model_id(metadata["model_id"])
    metadata["format_version"] = MODEL_FORMAT_VERSION
    metadata["component_model_ids"] = {}
    directory = Path(model_dir or HYBRID_MODEL_DIR) / metadata["model_id"]
    directory.mkdir(parents=True, exist_ok=False)
    for algorithm in algorithms:
        component = components[algorithm]
        component_metadata = {name: deepcopy(value) for name, value in metadata.items() if not name.startswith(("component_", "stacking_")) and name != "model_id"}
        component_metadata["config"] = replace(config, algorithm=algorithm).to_dict()
        component_metadata["evaluation_method"] = "single"
        component_metadata["evaluation_parameters"] = deepcopy(metadata["evaluation_parameters"][algorithm])
        component_metadata.update(component["training_details"])
        if algorithm == "mlp":
            saved = save_torch_model(component["model"], component["scaler"], component_metadata, directory / "components")
        else:
            saved = save_global_model(component["model"], component_metadata, directory / "components")
        metadata["component_model_ids"][algorithm] = saved["model_id"]
    stacking_model = result.get("stacking_model")
    validate_hybrid_metadata(metadata, stacking_model)
    if stacking_model is not None:
        joblib.dump(stacking_model, directory / "stacking.joblib", compress=3)
    temporary = directory / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(directory / "metadata.json")
    return metadata


def load_hybrid_model(model_id, model_dir=None):
    validate_model_id(model_id)
    if not HYBRID_MODEL_ID_PATTERN.fullmatch(model_id):
        raise ValueError("Invalid hybrid model ID")
    directory = Path(model_dir or HYBRID_MODEL_DIR) / model_id
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_id") != model_id:
        raise ValueError("Hybrid metadata does not match its ID")
    stacking_model = joblib.load(directory / "stacking.joblib") if metadata.get("evaluation_method") == "stacking" else None
    config = validate_hybrid_metadata(metadata, stacking_model)
    components = {}
    shared_fields = (
        "feature_columns", "training_cutoff", "train_tickers", "training_rows", "training_start_date",
        "training_end_date", "last_target_end_date", "news_policy", "news_start", "news_end",
        "market_policy", "sector_policy", "sector_benchmarks")
    for algorithm in metadata["component_algorithms"]:
        component_model, scaler, component_metadata = load_model(metadata["component_model_ids"][algorithm], directory / "components")
        expected_config = replace(config, algorithm=algorithm)
        if get_model_config(component_metadata) != expected_config or "calibration" in component_metadata:
            raise ValueError("Component configuration does not match the hybrid")
        if any(component_metadata.get(name) != metadata.get(name) for name in shared_fields):
            raise ValueError("Component data or feature policy does not match the hybrid")
        components[algorithm] = {
            "model": component_model,
            "scaler": scaler,
            "metadata": component_metadata,
            "training_details": deepcopy(metadata["component_training_details"][algorithm])}
    model = {"method": metadata["evaluation_method"], "model_results": components, "stacking_model": stacking_model}
    return model, None, metadata
