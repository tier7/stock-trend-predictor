import argparse


from services.ml.feature_schema import FEATURE_GROUPS
from services.ml.mlp_service import train_mlp_model
from services.ml.model_config import ModelConfig
from services.news_fetcher import NewsPreparationError
from services.ml.lightgbm_service import train_lightgbm_model

def main():
    parser = argparse.ArgumentParser(
        description="Wytrenuj i zapisz globalny model LightGBM.",
    )
    parser.add_argument(
        "--cutoff", required=True, metavar="YYYY-MM-DD",
        help="Data graniczna danych treningowych.",
    )
    parser.add_argument(
        "--validation-start", required=True, metavar="YYYY-MM-DD",
        help="Początek okresu walidacji.",
    )
    parser.add_argument(
        "--tickers", nargs="+", type=str.upper, required=True,
        help="Lista spółek treningowych.",
    )
    parser.add_argument(
        "--task", choices=("classification", "regression"),
        default="classification",
        help="Kierunek albo stopa zwrotu.",
    )
    parser.add_argument(
        "--horizon", type=int, default=1,
        help="Horyzont prognozy w sesjach giełdowych.",
    )
    parser.add_argument(
        "--feature-groups", nargs="+", choices=tuple(FEATURE_GROUPS),
        default=["technical"],
        help="Grupy cech używane przez model.",
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Liczba iteracji bez poprawy przed zatrzymaniem treningu.",
    )
    parser.add_argument(
        "--seed", type=int, default=26,
        help="Ziarno losowości modelu i treningu.",
    )
    parser.add_argument(
        "--news-start", metavar="YYYY-MM-DD",
        help="Początek zakresu newsów.",
    )
    parser.add_argument(
        "--news-end", metavar="YYYY-MM-DD",
        help="Koniec zakresu newsów, bez wskazanego dnia.",
    )
    parser.add_argument(
        "--prepared-news-only", action="store_true",
        help="Korzystaj wyłącznie z zapisanych newsów i ocen FinBERT.",
    )
    parser.add_argument(
        "--model-dir",
        help="Opcjonalny katalog zapisu modeli LightGBM.",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=500,
        help="Maksymalna liczba iteracji podczas wyboru modelu.",
    )

    arguments = parser.parse_args()
    try:
        if arguments.n_estimators < 1 or arguments.patience < 1:
            raise ValueError("n-estimators i patience muszą być dodatnie.")


        config = ModelConfig(
            algorithm="lightgbm",
            task=arguments.task,
            horizon=arguments.horizon,
            feature_groups=tuple(arguments.feature_groups),
            random_state=arguments.seed,
        )
        metadata = train_lightgbm_model(
            tickers=arguments.tickers,
            validation_start=arguments.validation_start,
            training_cutoff=arguments.cutoff,
            config=config,
            n_estimators=arguments.n_estimators,
            patience=arguments.patience,
            news_start=arguments.news_start,
            news_end=arguments.news_end,
            prepare_news=not arguments.prepared_news_only,
            model_dir=arguments.model_dir
        )
    except NewsPreparationError as error:
        parser.exit(1, f"Nie udało się przygotować newsów: {error}\n")
    except ValueError as error:
        parser.error(str(error))

    print(f"Zapisano model: {metadata['model_id']}")
    print(f"Zadanie: {config.task}")
    print(f"Horyzont: {config.horizon} sesji")
    print(f"Data graniczna treningu: {metadata['training_cutoff']}")
    print(f"Początek walidacji: {metadata['validation_start']}")
    print(f"Grupy cech: {', '.join(config.feature_groups)}")
    print(f"Liczba spółek: {len(metadata['train_tickers'])}")
    print(f"Wybrana liczba iteracji: {metadata['best_iteration']}")
    print(f"Wykonane iteracje treningu końcowego: {metadata['trained_iterations']}")


if __name__ == "__main__":
    main()
