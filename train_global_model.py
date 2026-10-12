"""Prepare a ready global model separately from running stock simulations."""

import argparse

from services.ml.global_forest_service import train_global_model


def main():
    parser = argparse.ArgumentParser(description="Wytrenuj i zapisz gotowy model globalny.")
    parser.add_argument(
        "--cutoff", required=True, metavar="YYYY-MM-DD",
        help="Data graniczna danych treningowych; symulacje będą możliwe od tej daty.",
    )
    arguments = parser.parse_args()
    try:
        metadata = train_global_model(arguments.cutoff, progress_callback=print)
    except ValueError as error:
        parser.error(str(error))
    print(f"Zapisano model: {metadata['model_id']}")
    print(f"Symulacja od: {metadata['training_cutoff']}")
    print(f"Aby wybrać tę wersję na stałe, ustaw GLOBAL_MODEL_ID={metadata['model_id']} w .env.")


if __name__ == "__main__":
    main()
