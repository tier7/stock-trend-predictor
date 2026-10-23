
import argparse

from services.ml.global_forest_service import train_global_model
from services.ml.model_config import ModelConfig

def main():
    parser = argparse.ArgumentParser(description="Wytrenuj i zapisz gotowy model globalny.")
    parser.add_argument(
        "--cutoff", required=True, metavar="YYYY-MM-DD",
        help="Data graniczna danych treningowych.",
    )

    parser.add_argument(
        "--task",
        choices=("classification", "regression"),
        default="classification",
        help="Kierunek albo stopa zwrotu.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Horyzont prognozy w sesjach giełdowych.",
    )

    arguments = parser.parse_args()
    try:
        config = ModelConfig(
            algorithm="random_forest",
            task=arguments.task,
            horizon=arguments.horizon,
        )

        metadata = train_global_model(
            arguments.cutoff,
            progress_callback=print,
            config=config,
        )
    except ValueError as error:
        parser.error(str(error))
    print(f"Zapisano model: {metadata['model_id']}")
    print(f"Zadanie: {config.task}")
    print(f"Horyzont: {config.horizon} sesji")
    print(f"Data graniczna treningu: {metadata['training_cutoff']}")



if __name__ == "__main__":
    main()
