"""SHAP, permutation importance, and the keep/drop audit."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

from points_xgb.paths import ensure_repo_on_path

ensure_repo_on_path()
from models.shared.metrics import pinball_50  # noqa: E402


def mean_absolute_shap(model, frame: pd.DataFrame) -> pd.Series:
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer(frame)
    importance = pd.Series(
        np.abs(values.values).mean(axis=0),
        index=frame.columns,
    )
    return importance.sort_values(ascending=True)


def plot_shap(importance: pd.Series):
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Mean |SHAP| — q_0.50")
    ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout()
    return fig


def permutation_importance_table(
    model,
    frame: pd.DataFrame,
    target,
    *,
    n_repeats: int = 30,
    seed: int = 42,
):
    """Val-set permutation importance. Higher means the shuffle hurt q50 pinball."""
    result = permutation_importance(
        model,
        frame,
        target,
        n_repeats=n_repeats,
        random_state=seed,
        scoring=make_scorer(pinball_50),
    )
    table = (
        pd.DataFrame(
            {
                "feature": frame.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    table.index += 1
    table["importance"] = table["importance"].round(5)
    table["std"] = table["std"].round(5)
    table["p_value"] = stats.ttest_1samp(result.importances, 0.0, axis=1).pvalue.round(
        4
    )
    table["significant"] = table["p_value"] < 0.05
    return table, result


def feature_consensus(
    features: list[str],
    shap_mean: pd.Series,
    permutation: pd.DataFrame,
    importances: np.ndarray,
    ablation: pd.DataFrame,
) -> pd.DataFrame:
    """Rank features by the average of normalized SHAP and permutation importance.

    ``keep`` requires a permutation p-value under 0.05 and a positive
    ablation delta. The p-value is recomputed with NaNs omitted, matching
    the notebook audit rather than the rounded table p-value.
    """
    shap_table = pd.DataFrame(
        {
            "feature": shap_mean.index,
            "mean_abs_shap": shap_mean.to_numpy(),
        }
    ).assign(
        shap_rank=lambda rows: rows["mean_abs_shap"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    perm_table = (
        permutation.sort_values("importance", ascending=False)
        .reset_index(drop=True)
        .rename(
            columns={"importance": "perm_importance", "std": "perm_importance_std"}
        )
    )
    perm_table["perm_rank"] = np.arange(1, len(perm_table) + 1)
    perm_table = perm_table[
        ["feature", "perm_rank", "perm_importance", "perm_importance_std"]
    ]
    if len(features) != importances.shape[0]:
        raise ValueError(
            f"{len(features)} features but importance matrix has "
            f"{importances.shape[0]} rows"
        )
    perm_pvalues = pd.DataFrame(
        {
            "feature": list(features),
            "perm_pvalue": stats.ttest_1samp(
                importances, 0.0, axis=1, nan_policy="omit"
            ).pvalue,
        }
    )
    dropped = ablation[["feature", "delta_pinball"]].rename(
        columns={"delta_pinball": "ablation_delta"}
    )
    audit = (
        pd.DataFrame({"feature": features})
        .merge(shap_table, on="feature", how="left")
        .merge(perm_table, on="feature", how="left")
        .merge(perm_pvalues, on="feature", how="left")
        .merge(dropped, on="feature", how="left")
    )
    shap_values = audit["mean_abs_shap"].astype(float)
    perm_values = audit["perm_importance"].astype(float)
    max_shap = shap_values.fillna(0).max()
    max_perm = perm_values.fillna(0).max()
    audit["shap_norm"] = np.where(max_shap > 0, shap_values.fillna(0) / max_shap, 0.0)
    audit["perm_norm"] = np.where(max_perm > 0, perm_values.fillna(0) / max_perm, 0.0)
    audit["consensus"] = (audit["shap_norm"] + audit["perm_norm"]) / 2
    audit = audit.sort_values("consensus", ascending=False).reset_index(drop=True)
    audit.index += 1
    audit["keep"] = (audit["perm_pvalue"] < 0.05) & (audit["ablation_delta"] > 0)
    return audit
