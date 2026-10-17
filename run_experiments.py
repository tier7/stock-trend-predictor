from services.ml.experiment_service import run_experiments
import pandas as pd
from pathlib import Path
from services.ml.experiment_service import run_experiments, summarize_results

output_path = Path(__file__).with_name("experiment_results.csv")
output_path_summary = Path(__file__).with_name("experiment_summary.csv")
periods = [
    {"name": "2026_Q1", "start_date": "2026-01-01", "end_date": "2026-03-31"},
    {"name": "2026_Q2", "start_date": "2026-04-01", "end_date": "2026-06-30"},
    {"name": "2026_Q3", "start_date": "2026-07-01", "end_date": "2026-09-30"},
]
tickers = [
    # Spółki obecne w treningu
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JNJ", "KO", "PG", "CAT", "BRK-B",

    # Spółki nieobecne w treningu
    "NVDA", "AMD", "JPM", "BAC", "PFE",
    "LLY", "XOM", "CVX", "VZ", "T",
]
model_types = ["global", "local", "mlp"]

if __name__ == "__main__":
    rows = run_experiments(
        tickers,
        model_types,
        periods,
        global_model_id="global_forest_2026-01-01_71b3dcc8",
        torch_model_id="mlp_2026-01-01_3f443296",
        confidence_threshold=0.6,
    )
    report = pd.DataFrame(rows)
    print(report)
    summary = summarize_results(report)
    print(summary.to_string(index=False))
    report.to_csv(output_path, index=False, encoding="utf-8")
    summary.to_csv(output_path_summary, index=False, encoding="utf-8")
