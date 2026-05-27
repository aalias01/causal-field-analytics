"""
src/estimators.py — Reusable causal and survival estimators.

Covers:
    1. Naive before/after comparison (straw man)
    2. Propensity score matching (ATT)
    3. Difference-in-Differences (OLS with fixed effects)
    4. Synthetic control (wrapper around pysyncon)
    5. Cox proportional hazards (hazard ratio + B10-life comparison)
    6. Weibull AFT (parametric reliability framing)
    7. Kaplan-Meier (non-parametric survival curves)

All estimators accept a processed DataFrame and return a results dict
with effect estimate, CI, and notes for the method-comparison table.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ─── 1. Naive comparison ─────────────────────────────────────────────────────

def naive_comparison(df: pd.DataFrame) -> dict:
    """
    Naive failure rate comparison: treated vs. control.
    Intentionally wrong — included to show confounding bias.
    """
    ctrl  = df[df.design_variant == "A"].failure_event.mean()
    treat = df[df.design_variant == "B"].failure_event.mean()
    effect_pct = (treat - ctrl) / ctrl * 100
    return {
        "method": "Naive before/after",
        "effect_estimate_pct": round(effect_pct, 2),
        "ci_lower": None, "ci_upper": None,
        "note": "Confounded by selection bias — newer units got variant B AND have lower baseline hazard",
    }


# ─── 2. Propensity Score Matching ────────────────────────────────────────────

def propensity_score_matching(df: pd.DataFrame, confounders: list[str]) -> dict:
    """
    Estimate ATT via nearest-neighbor propensity score matching.

    Returns matched DataFrame and ATT estimate.

    Note: This is a lightweight implementation; use `causalinference` library in notebook 03
    for full balance diagnostics and bootstrapped CIs.
    """
    df = df.copy()

    # Encode categorical confounders
    X = pd.get_dummies(df[confounders], drop_first=True)
    y = (df.design_variant == "B").astype(int)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, y)
    df["propensity_score"] = lr.predict_proba(X)[:, 1]

    treated = df[y == 1].copy()
    control = df[y == 0].copy()

    # Greedy nearest-neighbor match (1:1)
    control_idx = control.index.tolist()
    matched_control = []
    used = set()
    for _, t_row in treated.iterrows():
        diffs = abs(control.loc[[i for i in control_idx if i not in used], "propensity_score"] - t_row.propensity_score)
        if diffs.empty:
            continue
        best = diffs.idxmin()
        matched_control.append(control.loc[best])
        used.add(best)

    matched_df = pd.concat([treated, pd.DataFrame(matched_control)])
    att = matched_df.groupby("design_variant")["failure_event"].mean()
    effect_pct = (att.get("B", 0) - att.get("A", 0)) / att.get("A", 1) * 100

    return {
        "method": "Propensity Score Matching",
        "effect_estimate_pct": round(effect_pct, 2),
        "n_matched_pairs": len(matched_control),
        "note": "ATT on matched sample; CIs via bootstrap in notebook 03",
    }


# ─── 3. Difference-in-Differences ────────────────────────────────────────────

def difference_in_differences(
    df_panel: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    post_col: str,
    unit_col: str = "unit_id",
) -> dict:
    """
    DiD estimator using OLS with unit and time fixed effects.

    Args:
        df_panel: long-format panel with one row per unit-period
        outcome_col: outcome variable name
        treatment_col: 1 = treated group, 0 = control
        post_col: 1 = post-treatment period, 0 = pre
        unit_col: unit identifier for fixed effects

    Returns dict with ATT estimate, SE, p-value.
    """
    df = df_panel.copy()
    df["interaction"] = df[treatment_col] * df[post_col]

    formula = f"{outcome_col} ~ interaction + C({unit_col}) + C({post_col})"
    model = smf.ols(formula, data=df).fit(cov_type="HC1")

    coef  = model.params.get("interaction", np.nan)
    se    = model.bse.get("interaction", np.nan)
    pval  = model.pvalues.get("interaction", np.nan)
    ci    = model.conf_int().loc["interaction"].tolist() if "interaction" in model.conf_int().index else [None, None]

    return {
        "method": "Difference-in-Differences",
        "effect_estimate": round(coef, 4),
        "std_error": round(se, 4),
        "p_value": round(pval, 4),
        "ci_lower": round(ci[0], 4) if ci[0] else None,
        "ci_upper": round(ci[1], 4) if ci[1] else None,
        "note": "OLS with unit + time FE; HC1 robust SEs; verify parallel-trends assumption",
    }


# ─── 4. Cox Proportional Hazards ─────────────────────────────────────────────

def cox_ph(df: pd.DataFrame, confounders: list[str]) -> dict:
    """
    Cox PH regression for hazard ratio estimation.

    Args:
        df: panel with columns time_to_event, failure_event, design_variant, + confounders
        confounders: list of covariate column names

    Returns dict with hazard ratio, CI, p-value, and B10-life comparison.
    """
    from lifelines import CoxPHFitter

    df_surv = df.copy()
    df_surv["treatment"] = (df_surv.design_variant == "B").astype(int)

    # One-hot encode categorical confounders
    df_enc = pd.get_dummies(df_surv[["time_to_event", "failure_event", "treatment"] + confounders], drop_first=True)

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df_enc, duration_col="time_to_event", event_col="failure_event")

    summary = cph.summary
    hr_row  = summary.loc["treatment"] if "treatment" in summary.index else None

    hr  = round(float(hr_row["exp(coef)"]), 4)    if hr_row is not None else None
    ci_lower = round(float(hr_row["exp(coef) lower 95%"]), 4) if hr_row is not None else None
    ci_upper = round(float(hr_row["exp(coef) upper 95%"]), 4) if hr_row is not None else None
    pval = round(float(hr_row["p"]), 4) if hr_row is not None else None

    # B10-life (10th percentile of failure time) per group
    b10 = {}
    for variant, label in [("A", "control"), ("B", "treated")]:
        sub = df_surv[df_surv.design_variant == variant]
        if sub.failure_event.sum() > 10:
            from lifelines import KaplanMeierFitter
            kmf = KaplanMeierFitter()
            kmf.fit(sub.time_to_event, sub.failure_event)
            # B10 = time where 90% still survive (10% have failed)
            sf = kmf.survival_function_
            b10_idx = (sf["KM_estimate"] <= 0.90).idxmax()
            b10[label] = float(b10_idx) if b10_idx > 0 else None

    return {
        "method": "Cox Proportional Hazards",
        "hazard_ratio": hr,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": pval,
        "b10_life_control_months": b10.get("control"),
        "b10_life_treated_months": b10.get("treated"),
        "note": "Confounder-adjusted HR; verify proportional-hazards via Schoenfeld residuals",
    }


# ─── 5. Weibull AFT ──────────────────────────────────────────────────────────

def weibull_aft(df: pd.DataFrame) -> dict:
    """
    Weibull Accelerated Failure Time model per treatment group.
    Returns shape (rho) and scale (lambda) parameters + mean failure time.

    Interview framing: "The shape parameter tells us whether the failure rate
    is increasing (rho > 1 = wear-out) or decreasing (rho < 1 = burn-in).
    HVAC compressors should show rho > 1 after break-in period."
    """
    from lifelines import WeibullAFTFitter

    df_surv = df.copy()
    df_surv["treatment"] = (df_surv.design_variant == "B").astype(int)

    wf = WeibullAFTFitter()
    wf.fit(df_surv[["time_to_event", "failure_event", "treatment"]], duration_col="time_to_event", event_col="failure_event")

    return {
        "method": "Weibull AFT",
        "model": wf,
        "note": "Check Q-Q plot for goodness-of-fit; interpret shape parameter for wear-out vs. burn-in",
    }
