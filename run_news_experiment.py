import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from services.ml.global_forest_service import train_tickers, test_tickers
from services.ml.dataset_service import FEATURE_COLUMNS
from services.ml.news_feature_service import (
    NEWS_FEATURE_COLUMNS,
    build_market_schedule,
    prepare_dataset_with_news,
    prepare_dataset_with_news_timing,
)
from services.ml.finbert_service import MODEL_NAME, MODEL_REVISION
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from services.ml.walk_forward_service import split_dataset_by_date
import torch
from services.ml.mlp_model import NeuralNetwork
from services.ml.mlp_service import (prepare_tensors,create_data_loaders,train_model, calculate_baseline_loss, evaluate_model)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from torch.utils.data import TensorDataset, DataLoader


EXPERIMENT_SEEDS = [26, 42, 123]
EXCLUDED_TICKERS = {"MC.PA", "AIR.PA", "BRK-B", "META", "SHEL"}

def prepare_count_features(data, history_sessions=20):
    """Experimental features only; leave the shared dataset/cache unchanged.

    surprise = log(1 + today's count) - log(1 + mean of the previous 20 counts).
    Compute before splitting years so January can use the preceding December.
    No target or future count participates in the historical mean.
    """
    if not isinstance(history_sessions, int) or history_sessions < 1:
        raise ValueError("history_sessions must be a positive integer")
    result = data.sort_values(["ticker", "decision_at"]).reset_index(drop=True).copy()
    if result.duplicated(["ticker", "decision_at"]).any():
        raise ValueError("Duplicate ticker/decision time in count history")
    counts = result["preopen_news_count"]
    if not np.isfinite(counts.to_numpy(dtype=float)).all() or counts.lt(0).any():
        raise ValueError("News counts must be finite and nonnegative")
    result["preopen_log_count"] = np.log1p(counts)
    result["preopen_count_mean20"] = result.groupby("ticker")["preopen_news_count"].transform(
        lambda values: values.shift(1).rolling(history_sessions, min_periods=history_sessions).mean()
    )
    result["preopen_count_surprise20"] = (
        result["preopen_log_count"] - np.log1p(result["preopen_count_mean20"])
    )
    # Remove the same warmup rows from ALL variants, including the controls.
    return result.dropna(subset=["preopen_count_mean20"]).reset_index(drop=True)


def build_prediction_rows(test_df, evaluation, variant, test_group, seed, epochs,
                          baseline_probability, baseline_prediction):
    """Keep dates/prices for later backtests without fitting the model again."""
    context = [
        "ticker", "date", "feature_date", "execution_date", "next_date", "target_end_date",
        "decision_at", "market_open", "target_end_at", "next_open", "next_close", "target_end_open",
        "target", "preopen_news_count", "preopen_log_count",
        "preopen_count_mean20", "preopen_count_surprise20",
    ]
    rows = test_df[[column for column in context if column in test_df]].reset_index(drop=True).copy()
    rows["variant"] = variant
    rows["test_group"] = test_group
    rows["seed"] = seed
    rows["epochs"] = epochs
    rows["probability_up"] = evaluation["probabilities"]
    rows["prediction"] = evaluation["predictions"]
    rows["baseline_probability"] = baseline_probability
    rows["baseline_prediction"] = baseline_prediction
    return rows


def summarize_predictions_by_ticker(predictions):
    rows = []
    for (variant, group, seed, ticker), frame in predictions.groupby(
        ["variant", "test_group", "seed", "ticker"]
    ):
        model_loss = log_loss(frame["target"], frame["probability_up"], labels=[0, 1])
        baseline_loss = log_loss(frame["target"], frame["baseline_probability"], labels=[0, 1])
        rows.append({
            "variant": variant, "test_group": group, "seed": seed, "ticker": ticker,
            "test_rows": len(frame),
            "accuracy": float(frame["target"].eq(frame["prediction"]).mean()),
            "baseline_accuracy": float(frame["target"].eq(frame["baseline_prediction"]).mean()),
            "log_loss": model_loss, "baseline_log_loss": baseline_loss,
            "log_loss_gain": baseline_loss - model_loss,
            "news_coverage": float(frame["preopen_news_count"].gt(0).mean()),
            "mean_news_count": float(frame["preopen_news_count"].mean()),
        })
    return rows


def collect_news_dataset(tickers, start_date, end_date):
    datasets = []
    for ticker in tickers:
        dataset = prepare_dataset_with_news(ticker, start_date, end_date)
        datasets.append(dataset)
        print(ticker, len(dataset))

    return pd.concat(datasets, ignore_index=True)


def train_variant(train_df, validation_df, feature_columns, seed=26):
    torch.manual_seed(seed)
    x_train, y_train, x_validation, y_validation, scaler = prepare_tensors(train_df, validation_df, feature_columns)
    train_loader, validation_loader = create_data_loaders(x_train, y_train,x_validation, y_validation,)
    model = NeuralNetwork(input_size=len(feature_columns))
    model, history, best_epoch = train_model(model, train_loader, validation_loader)

    return model, scaler, history, best_epoch


def refit_variant(train_df, feature_columns, epochs, seed=26):
    torch.manual_seed(seed)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(train_df[feature_columns])
    x_train = torch.tensor(x_scaled, dtype=torch.float32)
    y_train = torch.tensor(
        train_df["target"].to_numpy(), dtype=torch.float32
    ).reshape(-1, 1)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = NeuralNetwork(input_size=len(feature_columns))
    model, history, _ = train_model(
        model,
        train_loader,
        validation_loader=None,
        epochs=epochs,
    )

    return model, scaler, history


def run_close_experiment(rebuild_cache=False):
    """The previous study, with splits based on the technical feature date."""

    train_symbols = [
        ticker for ticker in train_tickers if ticker not in EXCLUDED_TICKERS
    ]
    unseen_symbols = [
        ticker for ticker in test_tickers if ticker not in EXCLUDED_TICKERS
    ]

    cache_path = (
            Path(__file__).resolve().parent
            / "tmp"
            / "news_dataset_2021_2024_v1.pkl"
    )

    if cache_path.exists() and not rebuild_cache:
        data = pd.read_pickle(cache_path)
        print("Wczytano zapisany dataset")
    else:
        data = collect_news_dataset(
            train_symbols + unseen_symbols,
            "2021-01-01",
            "2025-01-01",
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_pickle(cache_path)
        print("Zapisano przygotowany dataset")

    known_df = data.loc[data["ticker"].isin(train_symbols)].copy()
    unseen_df = data.loc[data["ticker"].isin(unseen_symbols)].copy()

    train_df, validation_df = split_dataset_by_date(
        known_df,
        "2023-01-01",
        "2023-12-31",
    )

    _, seen_test_df = split_dataset_by_date(
        known_df,
        "2024-01-01",
        "2024-12-31",
    )

    _, unseen_test_df = split_dataset_by_date(
        unseen_df,
        "2024-01-01",
        "2024-12-31",
    )

    splits = {
        "train": train_df,
        "validation": validation_df,
        "seen_test": seen_test_df,
        "unseen_test": unseen_test_df,
    }

    for name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"Empty dataset: {name}")

        print(
            name,
            "Wiersze:", len(frame),
            "Spółki:", frame["ticker"].nunique(),
            "Od:", frame["date"].min(),
            "Do:", frame["date"].max(),
            "Ostatni target:", frame["target_end_date"].max(),
        )

    feature_columns = FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS

    print("Spółki treningowe:", len(train_symbols))
    print("Spółki odłożone:", len(unseen_symbols))
    print("Wszystkie spółki:", data["ticker"].nunique())
    print("Wiersze:", len(data))
    print("Braki:", data[feature_columns + ["target"]].isna().sum().sum())
    variants = {
        "technical": FEATURE_COLUMNS,
        "technical_news": FEATURE_COLUMNS + NEWS_FEATURE_COLUMNS,
        "technical_count": FEATURE_COLUMNS + ["news_count"]
    }

    final_train_df, _ = split_dataset_by_date(
        known_df,
        "2024-01-01",
        "2024-12-31",
    )
    print(
        "\nKońcowy trening 2021–2023:",
        "Wiersze:", len(final_train_df),
        "Spółki:", final_train_df["ticker"].nunique(),
    )

    test_groups = {
        "seen_test": seen_test_df,
        "unseen_test": unseen_test_df,
    }
    output_dir = (
        cache_path.parent
        / "news_experiments"
        / datetime.now(ZoneInfo("Europe/Warsaw")).strftime("%Y%m%d_%H%M%S_%f")
    )
    return run_comparison(
        train_df, validation_df, final_train_df, test_groups,
        variants, EXPERIMENT_SEEDS, output_dir,
    )


def run_comparison(train_df, validation_df, final_train_df, test_groups,
                   variants, experiment_seeds, output_dir, *, save_predictions=False):
    """Select epochs on 2023, refit on 2021-2023, evaluate each variant on 2024."""
    output_dir.mkdir(parents=True)
    baseline_loss = calculate_baseline_loss(train_df, validation_df)
    print(f"Validation baseline loss: {baseline_loss:.6f}")
    test_baselines = {
        name: calculate_baseline_loss(final_train_df, frame)
        for name, frame in test_groups.items()
    }
    baseline_prediction = int(final_train_df["target"].mode().iloc[0])
    baseline_probability = float(final_train_df["target"].mean())
    print("\nKatalog wyników:", output_dir)
    all_test_reports = []
    all_selection_results = []
    training_history = []
    ticker_results = []

    for experiment_seed in experiment_seeds:
        print(f"\nSeed: {experiment_seed}")
        selection_results = {}

        for variant_name, columns in variants.items():
            print(f"\nWariant: {variant_name}, liczba cech: {len(columns)}")

            model, scaler, history, best_epoch = train_variant(
                train_df,
                validation_df,
                columns,
                seed=experiment_seed,
            )

            evaluation = evaluate_model(
                model,
                validation_df,
                columns,
                scaler,
            )

            selection_results[variant_name] = {
                "best_epoch": best_epoch,
                "validation_log_loss": evaluation["log_loss"],
                "validation_accuracy": evaluation["accuracy"],
                "log_loss_gain": baseline_loss - evaluation["log_loss"],
            }
            all_selection_results.append({
                "seed": experiment_seed, "variant": variant_name,
                **selection_results[variant_name],
            })
            training_history.extend({
                "seed": experiment_seed, "variant": variant_name,
                "phase": "selection", **row,
            } for row in history)

        print(pd.DataFrame.from_dict(
            selection_results,
            orient="index",
        ).to_string())

        test_results = []
        seed_predictions = []
        for variant_name, columns in variants.items():
            epochs = selection_results[variant_name]["best_epoch"]
            print(f"\nKońcowy trening: {variant_name}, liczba epok: {epochs}")
            final_model, final_scaler, final_history = refit_variant(
                final_train_df,
                columns,
                epochs=epochs,
                seed=experiment_seed,
            )
            training_history.extend({
                "seed": experiment_seed, "variant": variant_name,
                "phase": "refit", **row,
            } for row in final_history)

            for test_name, test_df in test_groups.items():
                evaluation = evaluate_model(
                    final_model, test_df, columns, final_scaler
                )
                if save_predictions:
                    seed_predictions.append(build_prediction_rows(
                        test_df, evaluation, variant_name, test_name, experiment_seed, epochs,
                        baseline_probability, baseline_prediction,
                    ))
                test_results.append({
                    "variant": variant_name,
                    "test_group": test_name,
                    "seed": experiment_seed,
                    "features": len(columns),
                    "epochs": epochs,
                    "test_rows": len(test_df),
                    "accuracy": evaluation["accuracy"],
                    "baseline_accuracy": float(
                        (test_df["target"] == baseline_prediction).mean()
                    ),
                    "log_loss": evaluation["log_loss"],
                    "baseline_log_loss": test_baselines[test_name],
                    "log_loss_gain": test_baselines[test_name] - evaluation["log_loss"],
                })

        test_report = pd.DataFrame(test_results)
        all_test_reports.append(test_report)
        print(f"\nWyniki oceny na danych z 2024 roku, seed {experiment_seed}:")
        print(test_report.to_string(index=False))

        report_path = output_dir / f"news_experiment_2024_seed{experiment_seed}.csv"
        test_report.to_csv(report_path, index=False, encoding="utf-8")
        print("Zapisano wyniki:", report_path)
        if save_predictions:
            prediction_report = pd.concat(seed_predictions, ignore_index=True)
            prediction_path = output_dir / f"predictions_seed{experiment_seed}.csv.gz"
            prediction_report.to_csv(prediction_path, index=False, encoding="utf-8", compression="gzip")
            ticker_results.extend(summarize_predictions_by_ticker(prediction_report))
            print("Zapisano pojedyncze predykcje:", prediction_path)

    combined_report = pd.concat(all_test_reports, ignore_index=True)
    combined_report.to_csv(output_dir / "all_results.csv", index=False, encoding="utf-8")
    pd.DataFrame(all_selection_results).to_csv(
        output_dir / "validation_results.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(training_history).to_csv(
        output_dir / "training_history.csv", index=False, encoding="utf-8"
    )

    summary = combined_report.groupby(
        ["variant", "test_group"], as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_baseline_accuracy=("baseline_accuracy", "mean"),
        mean_log_loss=("log_loss", "mean"),
        std_log_loss=("log_loss", "std"),
        mean_baseline_log_loss=("baseline_log_loss", "mean"),
        mean_log_loss_gain=("log_loss_gain", "mean"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8")
    if ticker_results:
        ticker_report = pd.DataFrame(ticker_results)
        ticker_report.to_csv(output_dir / "ticker_results.csv", index=False, encoding="utf-8")
        ticker_summary = ticker_report.groupby(["variant", "test_group", "ticker"], as_index=False).agg(
            seeds=("seed", "nunique"), test_rows=("test_rows", "first"),
            mean_accuracy=("accuracy", "mean"), mean_baseline_accuracy=("baseline_accuracy", "mean"),
            mean_log_loss_gain=("log_loss_gain", "mean"), std_log_loss_gain=("log_loss_gain", "std"),
            positive_gain_seeds=("log_loss_gain", lambda values: int(values.gt(0).sum())),
            news_coverage=("news_coverage", "first"), mean_news_count=("mean_news_count", "first"),
        )
        ticker_summary.to_csv(output_dir / "ticker_summary.csv", index=False, encoding="utf-8")

    print("\nŚrednie wyniki i odchylenie standardowe między seedami:")
    print(summary.to_string(index=False))
    print("Zapisano raporty w:", output_dir)
    return combined_report, summary


def split_timing_period(data, start_date, end_date):
    """Half-open periods, purged using the actual time a target becomes known."""
    start = pd.to_datetime(start_date, utc=True)
    end = pd.to_datetime(end_date, utc=True)
    if start >= end:
        raise ValueError("Period start must precede period end")
    training = data.loc[
        data["decision_at"].lt(start) & data["target_end_at"].lt(start)
    ].copy()
    evaluation = data.loc[
        data["decision_at"].ge(start) & data["decision_at"].lt(end)
        & data["target_end_at"].lt(end)
    ].copy()
    return training, evaluation


def validate_timing_dataset(data, expected_tickers):
    required_features = FEATURE_COLUMNS + [
        f"{timing}_{column}"
        for timing in ["close", "preopen"] for column in NEWS_FEATURE_COLUMNS
    ]
    if data.empty or set(data["ticker"]) != set(expected_tickers):
        raise ValueError("The paired dataset must contain every requested ticker")
    if data.duplicated(["ticker", "execution_date"]).any():
        raise ValueError("Duplicate ticker/execution date in the paired dataset")
    if not np.isfinite(data[required_features].to_numpy(dtype=float)).all():
        raise ValueError("Missing or infinite features in the paired dataset")
    if not data["target"].isin([0, 1]).all():
        raise ValueError("Invalid target in the paired dataset")
    if not (data["close_window_start"].lt(data["previous_close"])
            & data["previous_close"].lt(data["decision_at"])
            & data["decision_at"].lt(data["market_open"])
            & data["market_open"].lt(data["target_end_at"])).all():
        raise ValueError("Invalid order of feature, decision, execution and target timestamps")
    for timing, lower, upper in [
        ("close", "close_window_start", "previous_close"),
        ("preopen", "previous_close", "decision_at"),
    ]:
        latest = data[f"{timing}_latest_available_at"]
        has_news = data[f"{timing}_news_count"].gt(0)
        if not (latest.notna().eq(has_news)
                & (~has_news | (latest.gt(data[lower]) & latest.le(data[upper])))).all():
            raise ValueError(f"News outside the allowed {timing} window")


def load_timing_dataset(cache_path, train_symbols, unseen_symbols, rebuild_cache=False):
    spec = {
        "schema_version": 1,
        "start_date": "2021-01-01", "end_date_exclusive": "2025-01-01",
        "calendar": "NYSE", "minutes_before_open": 5,
        "train_tickers": train_symbols, "unseen_tickers": unseen_symbols,
        "technical_features": FEATURE_COLUMNS, "news_features": NEWS_FEATURE_COLUMNS,
        "news_model": MODEL_NAME, "news_model_revision": MODEL_REVISION,
        "relevance": "single_ticker_nonempty_headline",
    }
    tickers = train_symbols + unseen_symbols
    if cache_path.exists() and not rebuild_cache:
        cached = pd.read_pickle(cache_path)
        if not isinstance(cached, dict) or cached.get("spec") != spec:
            raise ValueError("Incompatible timing cache; run with --rebuild-cache")
        data = cached["data"]
        print("Wczytano dataset porównania pór decyzji:", cache_path)
    else:
        schedule = build_market_schedule("2020-12-15", "2025-01-15", spec["calendar"])
        frames = []
        for position, ticker in enumerate(tickers, 1):
            frame = prepare_dataset_with_news_timing(
                ticker, spec["start_date"], spec["end_date_exclusive"],
                minutes_before_open=spec["minutes_before_open"], schedule=schedule,
            )
            frames.append(frame)
            print(f"[{position}/{len(tickers)}] {ticker}: {len(frame)} wspólnych sesji", flush=True)
        data = pd.concat(frames, ignore_index=True)
        validate_timing_dataset(data, tickers)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"spec": spec, "data": data}, cache_path)
        print("Zapisano dataset:", cache_path)
    validate_timing_dataset(data, tickers)
    return data, spec


def build_paired_differences(report, comparisons=None):
    if comparisons is None:
        comparisons = [
            ("preopen_count", "close_count"),
            ("preopen_news", "close_news"),
            ("preopen_news", "preopen_count"),
            ("preopen_news", "technical"),
        ]
    rows = []
    for (seed, group), frame in report.groupby(["seed", "test_group"]):
        indexed = frame.set_index("variant")
        for candidate, reference in comparisons:
            rows.append({
                "seed": seed, "test_group": group,
                "candidate": candidate, "reference": reference,
                "log_loss_improvement": float(
                    indexed.loc[reference, "log_loss"] - indexed.loc[candidate, "log_loss"]
                ),
                "accuracy_improvement": float(
                    indexed.loc[candidate, "accuracy"] - indexed.loc[reference, "accuracy"]
                ),
            })
    paired = pd.DataFrame(rows)
    summary = paired.groupby(["candidate", "reference", "test_group"], as_index=False).agg(
        seeds=("seed", "nunique"),
        mean_log_loss_improvement=("log_loss_improvement", "mean"),
        std_log_loss_improvement=("log_loss_improvement", "std"),
        improved_seeds=("log_loss_improvement", lambda values: int(values.gt(0).sum())),
        mean_accuracy_improvement=("accuracy_improvement", "mean"),
    )
    return paired, summary


def run_timing_experiment(rebuild_cache=False, suite="timing"):
    if suite not in {"count", "timing"}:
        raise ValueError("suite must be 'count' or 'timing'")
    train_symbols = [ticker for ticker in train_tickers if ticker not in EXCLUDED_TICKERS]
    unseen_symbols = [ticker for ticker in test_tickers if ticker not in EXCLUDED_TICKERS]
    cache_path = Path(__file__).resolve().parent / "tmp" / "news_timing_dataset_2021_2024_preopen5_v1.pkl"
    data, spec = load_timing_dataset(cache_path, train_symbols, unseen_symbols, rebuild_cache)
    if suite == "count":
        original_rows = len(data)
        data = prepare_count_features(data)
        validate_timing_dataset(data, train_symbols + unseen_symbols)
        print("Nowe cechy liczby newsów; pominięto", original_rows - len(data),
              "wierszy rozgrzewki (20 na spółkę) we wszystkich wariantach.")
    known = data.loc[data["ticker"].isin(train_symbols)]
    unseen = data.loc[data["ticker"].isin(unseen_symbols)]
    training, validation = split_timing_period(known, "2023-01-01", "2024-01-01")
    final_training, seen_test = split_timing_period(known, "2024-01-01", "2025-01-01")
    _, unseen_test = split_timing_period(unseen, "2024-01-01", "2025-01-01")
    splits = {
        "train": training, "validation": validation, "final_train": final_training,
        "seen_test": seen_test, "unseen_test": unseen_test,
    }
    coverage = []
    for name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"Empty dataset: {name}")
        print(name, "Wiersze:", len(frame), "Spółki:", frame["ticker"].nunique(),
              "Decyzje od:", frame["decision_at"].min(), "do:", frame["decision_at"].max(),
              "Ostatni target:", frame["target_end_at"].max())
        for ticker, ticker_frame in frame.groupby("ticker"):
            for timing in ["close", "preopen"]:
                counts = ticker_frame[f"{timing}_news_count"]
                coverage.append({
                    "split": name, "ticker": ticker, "window": timing,
                    "rows": len(ticker_frame), "sessions_with_news": int(counts.gt(0).sum()),
                    "news_coverage": float(counts.gt(0).mean()), "total_news": int(counts.sum()),
                })

    if suite == "count":
        variants = {
            "technical": FEATURE_COLUMNS,
            "preopen_count": FEATURE_COLUMNS + ["preopen_news_count"],
            "preopen_log_count": FEATURE_COLUMNS + ["preopen_log_count"],
            "preopen_log_surprise20": FEATURE_COLUMNS + ["preopen_log_count", "preopen_count_surprise20"],
        }
        comparisons = [
            ("preopen_count", "technical"),
            ("preopen_log_count", "preopen_count"),
            ("preopen_log_surprise20", "preopen_log_count"),
            ("preopen_log_surprise20", "preopen_count"),
        ]
    else:
        variants = {"technical": FEATURE_COLUMNS}
        for timing in ["close", "preopen"]:
            variants[f"{timing}_count"] = FEATURE_COLUMNS + [f"{timing}_news_count"]
            variants[f"{timing}_news"] = FEATURE_COLUMNS + [
                f"{timing}_{column}" for column in NEWS_FEATURE_COLUMNS
            ]
        comparisons = None
    print(f"Badanie: {suite} | warianty: {len(variants)} | seedy: {EXPERIMENT_SEEDS} | "
          f"wyniki testowe: {len(variants) * len(EXPERIMENT_SEEDS) * 2}")
    reports_folder = "news_count_experiments" if suite == "count" else "news_timing_experiments"
    output_dir = cache_path.parent / reports_folder / datetime.now(
        ZoneInfo("Europe/Warsaw")
    ).strftime("%Y%m%d_%H%M%S_%f")
    report, summary = run_comparison(
        training, validation, final_training,
        {"seen_test": seen_test, "unseen_test": unseen_test},
        variants, EXPERIMENT_SEEDS, output_dir,
        save_predictions=(suite == "count"),
    )
    paired, paired_summary = build_paired_differences(report, comparisons)
    paired.to_csv(output_dir / "paired_differences.csv", index=False, encoding="utf-8")
    paired_summary.to_csv(output_dir / "paired_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(coverage).to_csv(output_dir / "news_coverage.csv", index=False, encoding="utf-8")
    data.head(20)[[
        "ticker", "feature_date", "execution_date", "close_window_start", "previous_close",
        "decision_at", "market_open", "target_end_at", "target",
        "close_news_count", "close_latest_available_at",
        "preopen_news_count", "preopen_latest_available_at",
    ]].to_csv(output_dir / "timing_sample.csv", index=False, encoding="utf-8")
    manifest = {
        **spec, "suite": suite, "seeds": EXPERIMENT_SEEDS, "variants": variants,
        "cache_path": str(cache_path),
        "dataset_hash": hashlib.sha256(
            pd.util.hash_pandas_object(data, index=True).values.tobytes()
        ).hexdigest(),
        "pandas_version": pd.__version__, "torch_version": torch.__version__,
        "split_rows": {name: len(frame) for name, frame in splits.items()},
        "periods": {"train": "2021-2022", "validation": "2023", "test": "2024"},
        "target": "current session open to following session open",
        "close_window": "(close before feature session, feature session close]",
        "preopen_window": "(previous session close, current session open minus 5 minutes]",
        "all_variants_decision_at": "current session open minus 5 minutes",
        "availability": "max(created_at, updated_at) of each stored article version",
        "limitations": [
            "Historical API snapshots do not establish original reception times or full revision history.",
            "2024 has already been examined; this is a development comparison, not an untouched holdout.",
            "Seed variation is initialization variation on one market year, not independent market samples.",
            "Classification metrics do not measure net trading returns.",
        ],
    }
    if suite == "count":
        manifest["count_features"] = {
            "history_sessions": 20, "history_excludes_current_session": True,
            "log_count": "log1p(preopen_news_count)",
            "surprise": "log1p(current count) - log1p(mean of previous 20 counts for the same ticker)",
            "warmup": "first 20 rows per ticker removed from every variant; shared cache is unchanged",
        }
        manifest["prediction_files"] = "predictions_seed{seed}.csv.gz; pandas.read_csv reads gzip directly"
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nPorównania na tych samych seedach i sesjach; dodatnia poprawa = niższy log loss:")
    print(paired_summary.to_string(index=False))
    print("Pełne raporty:", output_dir)
    return report, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Porównanie cech technicznych i newsów FinBERT")
    parser.add_argument("--timing", choices=["preopen", "close"], default="preopen",
                        help="preopen: badanie wybrane przez --suite; close: pierwotny eksperyment")
    parser.add_argument("--suite", choices=["count", "timing"], default="count",
                        help="count (domyślnie): 4 warianty liczby newsów; timing: poprzednie 5 wariantów okien")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Przelicz dataset z danych w bazie, bez pobierania newsów ani uruchamiania FinBERT")
    args = parser.parse_args(argv)
    if args.timing == "close":
        run_close_experiment(rebuild_cache=args.rebuild_cache)
    else:
        run_timing_experiment(rebuild_cache=args.rebuild_cache, suite=args.suite)


if __name__ == "__main__":
    main()
