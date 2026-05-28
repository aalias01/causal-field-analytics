"""
Run the portfolio-ready analysis path for Causal Field Analytics.

This script generates the synthetic field panel, computes the headline
portfolio metrics, and saves the figures consumed by the Quarto report.
It intentionally favors transparent estimators over notebook-only state.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from src.data_generator import TRUE_HAZARD_RATIO, generate_panel


DATA_PATH = ROOT / "data" / "field_panel.csv"
FIG_DIR = ROOT / "figures"
METRICS_PATH = ROOT / "report" / "portfolio_metrics.json"
RNG = np.random.default_rng(42)


def pct(x: float) -> float:
    return round(float(x) * 100, 2)


def save_panel() -> pd.DataFrame:
    DATA_PATH.parent.mkdir(exist_ok=True)
    df = generate_panel()
    df.to_csv(DATA_PATH, index=False)
    df["install_date"] = pd.to_datetime(df["install_date"])
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["treatment"] = (out["design_variant"] == "B").astype(int)
    out["install_month_num"] = (
        (out["install_date"].dt.year - out["install_date"].dt.year.min()) * 12
        + out["install_date"].dt.month
    )
    out["install_quarter"] = out["install_date"].dt.to_period("Q").astype(str)
    out["post_rollout"] = (out["install_date"] >= "2023-01-01").astype(int)
    out["southeast_rollout"] = (out["region"] == "Southeast").astype(int)
    return out


def naive_result(df: pd.DataFrame) -> dict:
    rates = df.groupby("design_variant")["failure_event"].mean()
    diff_pp = pct(rates["B"] - rates["A"])
    rel_pct = pct((rates["B"] - rates["A"]) / rates["A"])
    return {
        "method": "Naive",
        "failure_rate_a": pct(rates["A"]),
        "failure_rate_b": pct(rates["B"]),
        "diff_pp": diff_pp,
        "relative_pct": rel_pct,
    }


def propensity_matching(df: pd.DataFrame, n_boot: int = 300) -> tuple[dict, pd.DataFrame]:
    features = [
        "region",
        "install_crew",
        "install_month_num",
        "operating_hours_month",
    ]
    x = pd.get_dummies(df[features], drop_first=True)
    y = df["treatment"]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x, y)
    scored = df.copy()
    scored["propensity_score"] = model.predict_proba(x)[:, 1]

    treated = scored[scored["treatment"] == 1].copy()
    control = scored[scored["treatment"] == 0].copy()
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[["propensity_score"]])
    _, idx = nn.kneighbors(treated[["propensity_score"]])
    matched_control = control.iloc[idx.ravel()].copy()
    matched = pd.concat([treated, matched_control], ignore_index=True)

    effect_pp = (
        treated["failure_event"].mean() - matched_control["failure_event"].mean()
    )
    boot = []
    n = len(treated)
    for _ in range(n_boot):
        sample_idx = RNG.integers(0, n, n)
        boot.append(
            treated.iloc[sample_idx]["failure_event"].mean()
            - matched_control.iloc[sample_idx]["failure_event"].mean()
        )
    ci = np.percentile(boot, [2.5, 97.5])

    return (
        {
            "method": "Propensity score matching",
            "effect_pp": pct(effect_pp),
            "ci_lower_pp": pct(ci[0]),
            "ci_upper_pp": pct(ci[1]),
            "n_matched_pairs": int(len(treated)),
        },
        matched,
    )


def difference_in_differences(df: pd.DataFrame) -> dict:
    model = smf.ols(
        "failure_event ~ southeast_rollout * post_rollout + C(region) + C(install_quarter)",
        data=df,
    ).fit(cov_type="HC1")
    term = "southeast_rollout:post_rollout"
    ci = model.conf_int().loc[term]
    return {
        "method": "Difference-in-differences",
        "effect_pp": pct(model.params[term]),
        "ci_lower_pp": pct(ci[0]),
        "ci_upper_pp": pct(ci[1]),
        "p_value": round(float(model.pvalues[term]), 4),
        "note": "Southeast rollout approximation with region and install-quarter fixed effects.",
    }


def synthetic_control(df: pd.DataFrame) -> dict:
    panel = (
        df.groupby(["region", "install_quarter"])["failure_event"]
        .mean()
        .unstack("region")
        .sort_index()
    )
    pre = panel.index < "2023Q1"
    controls = ["Southwest", "Midwest", "Northwest"]
    y_pre = panel.loc[pre, "Southeast"].to_numpy()
    x_pre = panel.loc[pre, controls].to_numpy()

    def objective(w: np.ndarray) -> float:
        return float(np.mean((y_pre - x_pre @ w) ** 2))

    result = minimize(
        objective,
        x0=np.repeat(1 / len(controls), len(controls)),
        bounds=[(0, 1)] * len(controls),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    weights = result.x
    synth = panel[controls].to_numpy() @ weights
    gap = panel["Southeast"].to_numpy() - synth
    post_gap = gap[~pre].mean()
    pre_rmspe = np.sqrt(np.mean(gap[pre] ** 2))

    return {
        "method": "Synthetic control",
        "post_gap_pp": pct(post_gap),
        "pre_rmspe_pp": pct(pre_rmspe),
        "weights": {c: round(float(w), 3) for c, w in zip(controls, weights)},
    }


def survival_models(df: pd.DataFrame) -> tuple[dict, CoxPHFitter, pd.DataFrame]:
    surv = df.copy()
    encoded = pd.get_dummies(
        surv[
            [
                "time_to_event",
                "failure_event",
                "treatment",
                "region",
                "install_crew",
                "install_month_num",
                "operating_hours_month",
            ]
        ],
        drop_first=True,
    )
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(encoded, duration_col="time_to_event", event_col="failure_event")
    row = cph.summary.loc["treatment"]

    reference = encoded.drop(columns=["time_to_event", "failure_event"]).mean().to_frame().T
    times = np.arange(1, 25)
    adjusted_b10 = {}
    for treatment, label in [(0, "A"), (1, "B")]:
        row_ref = reference.copy()
        row_ref["treatment"] = treatment
        sf = cph.predict_survival_function(row_ref, times=times).iloc[:, 0]
        below = sf[sf <= 0.90]
        adjusted_b10[label] = float(below.index.min()) if not below.empty else None

    a = surv[surv["design_variant"] == "A"]
    b = surv[surv["design_variant"] == "B"]
    lr = logrank_test(
        a["time_to_event"], b["time_to_event"], a["failure_event"], b["failure_event"]
    )
    ph = proportional_hazard_test(cph, encoded, time_transform="rank")
    treatment_ph_p = float(ph.summary.loc["treatment", "p"])

    aft = WeibullAFTFitter().fit(
        encoded[["time_to_event", "failure_event", "treatment"]],
        duration_col="time_to_event",
        event_col="failure_event",
    )
    aft_ratio = float(np.exp(aft.params_.loc[("lambda_", "treatment")]))

    return (
        {
            "method": "Cox proportional hazards",
            "hazard_ratio": round(float(row["exp(coef)"]), 3),
            "ci_lower": round(float(row["exp(coef) lower 95%"]), 3),
            "ci_upper": round(float(row["exp(coef) upper 95%"]), 3),
            "p_value": round(float(row["p"]), 4),
            "b10_a_months": adjusted_b10["A"],
            "b10_b_months": adjusted_b10["B"],
            "b10_delta_months": (
                round(adjusted_b10["B"] - adjusted_b10["A"], 1)
                if adjusted_b10["A"] is not None and adjusted_b10["B"] is not None
                else None
            ),
            "logrank_p_value": round(float(lr.p_value), 4),
            "cox_treatment_ph_p_value": round(treatment_ph_p, 4),
            "weibull_time_ratio": round(aft_ratio, 3),
        },
        cph,
        encoded,
    )


def plot_treatment_assignment(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    monthly = (
        df.assign(month=df["install_date"].dt.to_period("M").astype(str))
        .groupby("month")
        .agg(treatment_rate=("treatment", "mean"), failure_rate=("failure_event", "mean"))
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(monthly["month"], monthly["treatment_rate"], color="#1f77b4", marker="o")
    ax1.set_ylabel("Variant B share", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.tick_params(axis="x", rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(monthly["month"], monthly["failure_rate"], color="#d62728", marker="s")
    ax2.set_ylabel("Observed failure rate", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Treatment Assignment and Failure Rates Move Over Calendar Time")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "naive_confounders.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_propensity(matched: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.kdeplot(
        data=matched,
        x="propensity_score",
        hue="design_variant",
        common_norm=False,
        fill=True,
        alpha=0.35,
        ax=ax,
    )
    ax.set_title("Matched Propensity Score Overlap")
    ax.set_xlabel("Estimated propensity score")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "propensity_overlap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_parallel_trends(df: pd.DataFrame) -> None:
    quarterly = (
        df.groupby(["install_quarter", "region"])["failure_event"].mean().reset_index()
    )
    quarterly["rollout_group"] = np.where(
        quarterly["region"] == "Southeast", "Southeast rollout", "Other regions"
    )
    plot_df = (
        quarterly.groupby(["install_quarter", "rollout_group"])["failure_event"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.lineplot(
        data=plot_df,
        x="install_quarter",
        y="failure_event",
        hue="rollout_group",
        marker="o",
        ax=ax,
    )
    ax.axvline(3.5, color="black", linestyle="--", linewidth=1)
    ax.set_title("Parallel Trends Diagnostic for Southeast Rollout")
    ax.set_xlabel("Install quarter")
    ax.set_ylabel("Observed failure rate")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "parallel_trends.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_synthetic(df: pd.DataFrame, synth: dict) -> None:
    panel = (
        df.groupby(["region", "install_quarter"])["failure_event"]
        .mean()
        .unstack("region")
        .sort_index()
    )
    weights = synth["weights"]
    synthetic = sum(panel[col] * weight for col, weight in weights.items())
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(panel.index.astype(str), panel["Southeast"], marker="o", label="Southeast")
    ax.plot(panel.index.astype(str), synthetic, marker="s", label="Synthetic control")
    ax.axvline(3.5, color="black", linestyle="--", linewidth=1)
    ax.set_title("Southeast Failure Rate vs Synthetic Counterfactual")
    ax.set_xlabel("Install quarter")
    ax.set_ylabel("Observed failure rate")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synthetic_control.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_km(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"A": "#d62728", "B": "#1f77b4"}
    for variant in ["A", "B"]:
        sub = df[df["design_variant"] == variant]
        kmf = KaplanMeierFitter().fit(
            sub["time_to_event"], sub["failure_event"], label=f"Variant {variant}"
        )
        kmf.plot_survival_function(ax=ax, color=palette[variant], ci_show=True)
    ax.axhline(0.90, color="gray", linestyle=":", label="B10 threshold")
    ax.set_title("Kaplan-Meier Survival Curves")
    ax.set_xlabel("Months since install")
    ax.set_ylabel("Survival probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "kaplan_meier.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_forest(metrics: dict) -> None:
    rows = [
        ("Naive relative failure-rate change", metrics["naive"]["relative_pct"], None, None),
        (
            "PSM failure-rate difference (pp)",
            metrics["psm"]["effect_pp"],
            metrics["psm"]["ci_lower_pp"],
            metrics["psm"]["ci_upper_pp"],
        ),
        (
            "DiD failure-rate difference (pp)",
            metrics["did"]["effect_pp"],
            metrics["did"]["ci_lower_pp"],
            metrics["did"]["ci_upper_pp"],
        ),
        ("Synthetic control gap (pp)", metrics["synthetic_control"]["post_gap_pp"], None, None),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(rows))[::-1]
    for i, (label, estimate, low, high) in enumerate(rows):
        color = "#1f77b4" if estimate < 0 else "#d62728"
        ax.scatter(estimate, y[i], color=color, s=80, zorder=3)
        if low is not None and high is not None:
            ax.hlines(y[i], low, high, color=color, linewidth=2)
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Effect estimate")
    ax.set_title(
        f"Method Comparison: Cox HR = {metrics['survival']['hazard_ratio']} "
        f"(true HR = {TRUE_HAZARD_RATIO})"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "method_comparison_forest.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    df = add_features(save_panel())
    naive = naive_result(df)
    psm, matched = propensity_matching(df)
    did = difference_in_differences(df)
    synth = synthetic_control(df)
    survival, _, _ = survival_models(df)

    metrics = {
        "dataset": {
            "n_units": int(len(df)),
            "n_variant_a": int((df["design_variant"] == "A").sum()),
            "n_variant_b": int((df["design_variant"] == "B").sum()),
            "true_hazard_ratio": TRUE_HAZARD_RATIO,
            "true_hazard_reduction_pct": pct(1 - TRUE_HAZARD_RATIO),
        },
        "naive": naive,
        "psm": psm,
        "did": did,
        "synthetic_control": synth,
        "survival": survival,
    }

    plot_treatment_assignment(df)
    plot_propensity(matched)
    plot_parallel_trends(df)
    plot_synthetic(df, synth)
    plot_km(df)
    plot_forest(metrics)

    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
