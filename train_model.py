import argparse
import math

from services.ml.feature_schema import FEATURE_GROUPS
from services.ml.global_forest_service import train_global_model
from services.ml.lightgbm_service import train_lightgbm_model
from services.ml.mlp_service import train_mlp_model
from services.ml.model_config import ModelConfig
from services.ml.sector_feature_service import load_sector_assignments
from services.news_fetcher import NewsPreparationError


def main():
    parser = argparse.ArgumentParser(description="Wytrenuj i zapisz globalny model RF, MLP lub LightGBM.")
    parser.add_argument("--algorithm", required=True, choices=("random_forest", "mlp", "lightgbm"), help="Algorytm modelu.")
    parser.add_argument("--cutoff", required=True, metavar="YYYY-MM-DD", help="Data graniczna danych treningowych.")
    parser.add_argument("--validation-start", metavar="YYYY-MM-DD", help="Początek walidacji; wymagany dla MLP i LightGBM.")
    parser.add_argument("--tickers", nargs="+", type=str.upper, help="Spółki treningowe; wymagane dla MLP i LightGBM. RF domyślnie używa swojej listy.")
    parser.add_argument("--task", choices=("classification", "regression"), default="classification", help="Kierunek albo stopa zwrotu.")
    parser.add_argument("--horizon", type=int, default=1, help="Horyzont prognozy w sesjach giełdowych.")
    parser.add_argument("--feature-groups", nargs="+", choices=tuple(FEATURE_GROUPS), default=["technical"], help="Grupy cech używane przez model.")
    parser.add_argument("--seed", type=int, default=26, help="Ziarno losowości modelu i treningu.")
    parser.add_argument("--epochs", type=int, help="Maksymalna liczba epok MLP; domyślnie 50.")
    parser.add_argument("--learning-rate", type=float, help="Współczynnik uczenia Adam dla MLP; domyślnie 0.001.")
    parser.add_argument("--batch-size", type=int, help="Liczba próbek w partii MLP; domyślnie 256.")
    parser.add_argument("--patience", type=int, help="Liczba kroków bez poprawy; domyślnie 5 epok MLP lub 20 iteracji LightGBM.")
    parser.add_argument("--n-estimators", type=int, help="Maksymalna liczba iteracji LightGBM; domyślnie 500.")
    parser.add_argument("--news-start", metavar="YYYY-MM-DD", help="Początek zakresu newsów.")
    parser.add_argument("--news-end", metavar="YYYY-MM-DD", help="Koniec zakresu newsów, bez wskazanego dnia.")
    parser.add_argument("--prepared-news-only", action="store_true", help="Korzystaj wyłącznie z zapisanych newsów i ocen FinBERT.")
    parser.add_argument("--sector-map", metavar="PATH", help="Plik JSON z przypisaniami ticker -> sektor dla grupy sector.")
    parser.add_argument("--model-dir", help="Opcjonalny katalog zapisu modeli wybranego algorytmu.")

    arguments = parser.parse_args()
    try:
        sector_assignments = load_sector_assignments(arguments.sector_map)
        config = ModelConfig(algorithm=arguments.algorithm, task=arguments.task, horizon=arguments.horizon, feature_groups=tuple(arguments.feature_groups), random_state=arguments.seed)

        if "sector" in config.feature_groups and sector_assignments is None:
            raise ValueError("Grupa sector wymaga --sector-map PATH")
        if config.algorithm != "random_forest":
            if arguments.validation_start is None or not arguments.tickers:
                raise ValueError("MLP i LightGBM wymagają --validation-start i --tickers")
        elif arguments.validation_start is not None or arguments.patience is not None:
            raise ValueError("--validation-start i --patience dotyczą MLP oraz LightGBM")
        if config.algorithm != "mlp" and any(value is not None for value in (arguments.epochs, arguments.learning_rate, arguments.batch_size)):
            raise ValueError("--epochs, --learning-rate i --batch-size dotyczą wyłącznie MLP")
        if config.algorithm != "lightgbm" and arguments.n_estimators is not None:
            raise ValueError("--n-estimators dotyczy wyłącznie LightGBM")

        training_options = {
            "config": config,
            "news_start": arguments.news_start,
            "news_end": arguments.news_end,
            "prepare_news": not arguments.prepared_news_only,
            "sector_assignments": sector_assignments,
            "model_dir": arguments.model_dir
        }

        if config.algorithm == "random_forest":
            metadata = train_global_model(arguments.cutoff, training_tickers=arguments.tickers, progress_callback=print, **training_options)
        elif config.algorithm == "mlp":
            epochs = 50 if arguments.epochs is None else arguments.epochs
            learning_rate = 0.001 if arguments.learning_rate is None else arguments.learning_rate
            batch_size = 256 if arguments.batch_size is None else arguments.batch_size
            patience = 5 if arguments.patience is None else arguments.patience
            if epochs < 1 or batch_size < 1 or patience < 1:
                raise ValueError("epochs, batch-size i patience muszą być dodatnie")
            if not math.isfinite(learning_rate) or learning_rate <= 0:
                raise ValueError("learning-rate musi być skończoną, dodatnią liczbą")
            metadata = train_mlp_model(arguments.tickers, arguments.validation_start, arguments.cutoff, epochs=epochs, learning_rate=learning_rate, patience=patience, batch_size=batch_size, **training_options)
        else:
            n_estimators = 500 if arguments.n_estimators is None else arguments.n_estimators
            patience = 20 if arguments.patience is None else arguments.patience
            if n_estimators < 1 or patience < 1:
                raise ValueError("n-estimators i patience muszą być dodatnie")
            metadata = train_lightgbm_model(arguments.tickers, arguments.validation_start, arguments.cutoff, n_estimators=n_estimators, patience=patience, **training_options)
    except NewsPreparationError as error:
        parser.exit(1, f"Nie udało się przygotować newsów: {error}\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Zapisano model: {metadata['model_id']}")
    print(f"Algorytm: {config.algorithm}")
    print(f"Zadanie: {config.task}")
    print(f"Horyzont: {config.horizon} sesji")
    print(f"Data graniczna treningu: {metadata['training_cutoff']}")
    print(f"Grupy cech: {', '.join(config.feature_groups)}")
    print(f"Liczba spółek: {len(metadata['train_tickers'])}")
    if config.algorithm != "random_forest":
        print(f"Początek walidacji: {metadata['validation_start']}")
    if config.algorithm == "mlp":
        print(f"Liczba epok treningu końcowego: {metadata['epochs']}")
    elif config.algorithm == "lightgbm":
        print(f"Wybrana liczba iteracji: {metadata['best_iteration']}")
        print(f"Wykonane iteracje treningu końcowego: {metadata['trained_iterations']}")


if __name__ == "__main__":
    main()
