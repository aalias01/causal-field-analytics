"""
src/diagnostics.py — Diagnostic plots and statistical tests for causal + survival methods.

Functions:
    plot_propensity_balance(df_before, df_after, confounders)   — covariate balance check
    plot_parallel_trends(df_panel, treatment_col, outcome_col)  — DiD assumption visual
    plot_kaplan_meier(df, group_col)                            — KM survival curves
    plot_hazard_ratio_forest(results_list)                      — method comparison
    schoenfeld_test(cph_model)                                  — proportional hazards test
    log_rank_test(df, group_col)                                — KM significance test
    weibull_qq_plot(df, group_col)                              — Weibull goodness-of-fit
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def plot_propensity_balance(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    confounders: list[str],
    treatment_col: str = "design_variant",
    save_path: str = None,
) -> plt.Figure:
    """
    Standardized mean difference (SMD) plot for covariate balance before and after matching.
    Target: SMD < 0.1 after matching for all confounders.
    """
    smds_before, smds_after = [], []
    for col in confounders:
        if df_before[col].dtype == "object":
            continue
        for label, smds, df_ in [("Before", smds_before, df_before), ("After", smds_after, df_after)]:
            t = df_[df_[treatment_col] == "B"][col]
            c = df_[df_[treatment_col] == "A"][col]
            pooled_std = np.sqrt((t.std()**2 + c.std()**2) / 2)
            smd = abs(t.mean() - c.mean()) / (pooled_std + 1e-10)
            smds.append(smd)

    fig, ax = plt.subplots(figsize=(7, max(3, len(confounders))))
    numeric_conf = [c for c in confounders if df_before[c].dtype != "object"]
    y = np.arange(len(numeric_conf))
    ax.scatter(smds_before, y, label="Before matching", color="tomato", s=60, zorder=3)
    ax.scatter(smds_after,  y, label="After matching",  color="steelblue", s=60, zorder=3)
    ax.axvline(0.1, color="gray", linestyle="--", alpha=0.7, label="SMD = 0.10 threshold")
    ax.set_yticks(y); ax.set_yticklabels(numeric_conf)
    ax.set_xlabel("Standardized Mean Difference")
    ax.set_title("Covariate Balance: Before vs. After Propensity Matching")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_parallel_trends(
    df_panel: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    time_col: str,
    treatment_start: float,
    save_path: str = None,
) -> plt.Figure:
    """
    Parallel-trends diagnostic plot for DiD.
    Pre-treatment trajectories should be parallel; post-treatment divergence = causal effect.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, color in [("B", "steelblue"), ("A", "tomato")]:
        sub = df_panel[df_panel[treatment_col] == label].groupby(time_col)[outcome_col].mean()
        ax.plot(sub.index, sub.values, marker="o", label=f"Variant {label}", color=color, linewidth=2)

    ax.axvline(treatment_start, color="gold", linestyle="--", linewidth=1.5, label="Treatment start")
    ax.set_xlabel(time_col); ax.set_ylabel(f"Mean {outcome_col}")
    ax.set_title("Parallel Trends Diagnostic — Pre-treatment trajectories should overlap")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_kaplan_meier(
    df: pd.DataFrame,
    group_col: str = "design_variant",
    duration_col: str = "time_to_event",
    event_col: str = "failure_event",
    save_path: str = None,
) -> plt.Figure:
    """
    Kaplan-Meier survival curves for two groups with log-rank test p-value.
    Also shows B10-life (10% cumulative failure) reference line.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = df[group_col].unique()
    colors = {"A": "tomato", "B": "steelblue"}
    kmfs = {}
    for grp in sorted(groups):
        sub = df[df[group_col] == grp]
        kmf = KaplanMeierFitter()
        kmf.fit(sub[duration_col], sub[event_col], label=f"Variant {grp}")
        kmf.plot_survival_function(ax=ax, color=colors.get(grp, "gray"), ci_show=True)
        kmfs[grp] = kmf

    # Log-rank test
    if len(groups) == 2:
        g0, g1 = sorted(groups)
        s0 = df[df[group_col] == g0]
        s1 = df[df[group_col] == g1]
        result = logrank_test(s0[duration_col], s1[duration_col], s0[event_col], s1[event_col])
        ax.set_title(f"Kaplan-Meier Survival Curves\nLog-rank test p = {result.p_value:.4f}")
    else:
        ax.set_title("Kaplan-Meier Survival Curves")

    ax.axhline(0.90, color="gray", linestyle=":", alpha=0.6, label="B10 threshold (10% failure)")
    ax.set_xlabel("Time (months)"); ax.set_ylabel("Survival Probability")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_hazard_ratio_forest(
    results: list[dict],
    true_hr: float = 0.85,
    save_path: str = None,
) -> plt.Figure:
    """
    Forest plot comparing effect estimates across all methods.
    For causal methods: effect in % change in failure rate.
    For Cox PH: hazard ratio (log scale).
    Reference line = true causal effect.
    """
    fig, ax = plt.subplots(figsize=(9, max(4, len(results))))
    y_pos = list(range(len(results)))[::-1]

    for i, res in enumerate(results):
        y = y_pos[i]
        label   = res.get("method", "")
        est     = res.get("effect_estimate_pct") or res.get("hazard_ratio")
        ci_low  = res.get("ci_lower")
        ci_high = res.get("ci_upper")
        color   = "steelblue" if est and est < 0 else "tomato"

        if est is not None:
            ax.scatter([est], [y], color=color, s=80, zorder=3)
            if ci_low is not None and ci_high is not None:
                ax.hlines(y, ci_low, ci_high, color=color, linewidth=2)
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=9, transform=ax.get_yaxis_transform())

    if true_hr:
        ax.axvline(true_hr, color="gold", linestyle="--", linewidth=1.5, label=f"True HR = {true_hr}")
    ax.axvline(1.0, color="gray", linestyle="-", alpha=0.4, label="No effect")
    ax.set_yticks([]); ax.set_xlabel("Effect Estimate (HR or % change)")
    ax.set_title("Method Comparison Forest Plot")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def schoenfeld_test(cph_model) -> pd.DataFrame:
    """
    Test the proportional hazards assumption for a fitted CoxPHFitter.
    Returns DataFrame of test statistics and p-values.
    Flag variables with p < 0.05 as PH assumption potentially violated.
    """
    return cph_model.check_assumptions(training_df=None, p_value_threshold=0.05, show_plots=False)


def log_rank_test(
    df: pd.DataFrame,
    group_col: str = "design_variant",
    duration_col: str = "time_to_event",
    event_col: str = "failure_event",
) -> dict:
    """Log-rank test for equality of survival curves."""
    from lifelines.statistics import logrank_test
    groups = sorted(df[group_col].unique())
    if len(groups) != 2:
        return {"error": "Log-rank test requires exactly 2 groups"}
    g0, g1 = groups
    s0 = df[df[group_col] == g0]
    s1 = df[df[group_col] == g1]
    result = logrank_test(s0[duration_col], s1[duration_col], s0[event_col], s1[event_col])
    return {
        "test_statistic": round(result.test_statistic, 4),
        "p_value": round(result.p_value, 6),
        "significant": result.p_value < 0.05,
    }
