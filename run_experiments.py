from services.ml.experiment_service import run_experiments
import pandas as pd
from pathlib import Path
from services.ml.experiment_service import run_experiments, summarize_results

output_path = Path(__file__).with_name("experiment_results.csv")
output_path_summary = Path(__file__).with_name("experiment_results.csv")

periods = [
    {"name": "2026_Q1", "start_date": "2026-01-01", "end_date": "2026-03-31"},
    {"name": "2026_Q2", "start_date": "2026-04-01", "end_date": "2026-06-30"},
    {"name": "2026_Q3", "start_date": "2026-07-01", "end_date": "2026-09-30"},
]
tickers = ["AMD", "GOOGL"]
model_types = ["global", "local"]

if __name__ == "__main__":
    #rows = run_experiments(tickers, model_types, periods, global_model_id="global_forest_2026-01-01_71b3dcc8")
    #report = pd.DataFrame(rows)
    report = pd.read_csv(output_path)
    print(report)
    summary = summarize_results(report)
    print(summary.to_string(index=False))
    report.to_csv(output_path, index=False, encoding="utf-8")
    report.to_csv(output_path_summary, index=False, encoding="utf-8")
