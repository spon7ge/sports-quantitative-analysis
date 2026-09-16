"""Daily strikeout prediction reports (CSV + markdown/text)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.mlb.schemas import PREDICTION_COLUMNS, coerce_frame


def _line_prob_columns(frame: pd.DataFrame) -> list[str]:
    return [
        c for c in frame.columns if c.startswith("p_over_") or c.startswith("p_under_")
    ]


def write_daily_report(predictions: pd.DataFrame, path: str | Path) -> Path:
    """Write prediction CSV plus a markdown report of E[K] and O/U probs."""
    dest = Path(path)
    if dest.suffix.lower() in {".md", ".txt"}:
        report_path = dest
        csv_path = dest.with_suffix(".csv")
    elif dest.suffix.lower() == ".csv":
        csv_path = dest
        report_path = dest.with_suffix(".md")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        csv_path = dest / "predictions.csv"
        report_path = dest / "report.md"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ordered = coerce_frame(predictions, PREDICTION_COLUMNS)
        extra = [c for c in predictions.columns if c not in ordered.columns]
        if extra:
            export = pd.concat([ordered, predictions[extra]], axis=1)
        else:
            export = ordered
    except Exception:
        export = predictions.copy()
    export.to_csv(csv_path, index=False)

    prob_cols = _line_prob_columns(predictions)[:6]
    header = ["pitcher_id", "game_pk", "expected_k", *prob_cols]
    lines = [
        "# MLB starter strikeout daily report",
        "",
        "Hobby forecast. Not gambling advice. Over/under probabilities are model",
        "outputs, not implied profitability.",
        "",
        f"Rows: {len(predictions)}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    show = predictions.copy()
    if "expected_k" in show.columns:
        show = show.sort_values("expected_k", ascending=False)
    for _, row in show.head(50).iterrows():
        pitcher = row["pitcher_id"] if "pitcher_id" in row else ""
        game = row["game_pk"] if "game_pk" in row else ""
        expected = row["expected_k"] if "expected_k" in row else ""
        try:
            expected_txt = f"{float(expected):.2f}"
        except (TypeError, ValueError):
            expected_txt = str(expected)
        probs = []
        for col in prob_cols:
            try:
                probs.append(f"{float(row[col]):.3f}")
            except (TypeError, ValueError):
                probs.append("")
        lines.append(
            f"| {pitcher} | {game} | {expected_txt} | " + " | ".join(probs) + " |"
        )
    if (
        "market_disagreement" in predictions.columns
        and predictions["market_disagreement"].notna().any()
    ):
        lines.extend(
            [
                "",
                "Market disagreement is model_p_over minus de-vigged market_p_over.",
                "Quoted-price simulation is not realized ROI.",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n")
    return report_path
