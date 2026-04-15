from pathlib import Path
from datetime import datetime
import pandas as pd


REPORT_COLUMNS = [
    "timestamp",
    "task",
    "mode",
    "model",
    "train_csv",
    "eval_csv",
    "cv",
    "sample_size",
    "precision",
    "recall",
    "f1",
    "precision_macro_mean",
    "recall_macro_mean",
    "f1_macro_mean",
    "notes",
]


def save_eval_result(record: dict, reports_dir="reports"):
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **{col: record.get(col, "") for col in REPORT_COLUMNS if col != "timestamp"},
    }

    csv_path = reports_dir / "eval_results.csv"
    txt_path = reports_dir / "latest_eval.txt"

    df = pd.DataFrame([row], columns=REPORT_COLUMNS)

    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(txt_path, "w", encoding="utf-8") as f:
        for col in REPORT_COLUMNS:
            f.write(f"{col}: {row.get(col, '')}\n")
