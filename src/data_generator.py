"""
src/data_generator.py — Synthetic field-failure panel with known causal ground truth.

Generates ~5,000 equipment units observed over 24 months.
Treatment (design variant B) assigned non-randomly — newer units more likely
to receive the new design, creating realistic selection bias.

TRUE CAUSAL EFFECT: variant B reduces failure hazard by 15% (HR = 0.85).

Each record:
    unit_id, install_date, region, install_crew, design_variant,
    time_to_event (months), failure_event (0/1), operating_hours_per_month

The panel contains realistic confounders:
    - Seasonality: HVAC fails more in hot months (month 6–9)
    - Regional confounding: Southeast has higher ambient temperature → higher baseline hazard
    - Selection bias: newer units (post Jan 2023) more likely to receive variant B
    - Install-crew quality: crew A/B/C have different defect-introduction rates

Usage:
    python src/data_generator.py
    # Saves: data/field_panel.csv
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

N_UNITS = 5000
OBS_WINDOW_MONTHS = 24
START_DATE = date(2022, 1, 1)
END_DATE   = date(2023, 12, 31)
ANALYSIS_END = date(2024, 12, 31)

# ─── True causal parameters ───────────────────────────────────────────────────
TRUE_HAZARD_RATIO = 0.85      # variant B reduces hazard by 15%
BASELINE_MONTHLY_HAZARD = 0.015  # ~15% of units fail over 24 months naively

# Regional hazard multipliers (confounders)
REGION_HAZARD = {
    "Southeast":  1.25,   # hot climate → more stress → more failures
    "Southwest":  1.10,
    "Midwest":    0.95,
    "Northwest":  0.85,
}

# Install crew defect rates (confounders)
CREW_HAZARD = {
    "Crew_A": 0.90,   # skilled, low defect rate
    "Crew_B": 1.00,   # average
    "Crew_C": 1.20,   # less experienced, higher defect rate
}

# Seasonal multiplier by month (peak in summer)
SEASONAL = {1:0.7, 2:0.75, 3:0.85, 4:0.90, 5:1.05, 6:1.20,
            7:1.30, 8:1.25, 9:1.10, 10:0.90, 11:0.75, 12:0.70}

OUT_PATH = Path("data/field_panel.csv")


def _assign_treatment(install_date: date) -> str:
    """
    Non-random treatment assignment: newer units more likely to get variant B.
    Creates realistic selection bias — naive comparison is confounded.
    """
    cutoff = date(2023, 1, 1)
    months_after_cutoff = max(0, (install_date - cutoff).days / 30)
    p_treatment = 0.15 + min(0.70, months_after_cutoff * 0.06)
    return "B" if rng.random() < p_treatment else "A"


def _simulate_tte(
    region: str,
    crew: str,
    install_date: date,
    variant: str,
    obs_end: date,
) -> tuple[float, int]:
    """
    Simulate time-to-event (months) using a discrete-time hazard model.
    Returns (duration, event_observed).
    """
    hr_treatment = TRUE_HAZARD_RATIO if variant == "B" else 1.0
    region_mult  = REGION_HAZARD[region]
    crew_mult    = CREW_HAZARD[crew]

    max_months = min(OBS_WINDOW_MONTHS, round((obs_end - install_date).days / 30))
    if max_months <= 0:
        return 0.0, 0

    for month in range(1, max_months + 1):
        obs_month = (install_date.month - 1 + month - 1) % 12 + 1
        seasonal  = SEASONAL[obs_month]
        age_mult  = 1.0 + 0.01 * month  # slight increasing hazard with age

        h = BASELINE_MONTHLY_HAZARD * hr_treatment * region_mult * crew_mult * seasonal * age_mult
        h = min(h, 0.5)  # cap to avoid numerical issues

        if rng.random() < h:
            return float(month), 1

    # Right-censored — survived the observation window
    return float(max_months), 0


def generate_panel(n: int = N_UNITS) -> pd.DataFrame:
    date_range = (END_DATE - START_DATE).days
    regions   = list(REGION_HAZARD.keys())
    crews     = list(CREW_HAZARD.keys())
    region_probs = [0.30, 0.25, 0.30, 0.15]   # Southeast is largest market

    records = []
    for i in range(n):
        install_date = START_DATE + timedelta(days=int(rng.integers(0, date_range)))
        region       = rng.choice(regions, p=region_probs)
        crew         = rng.choice(crews)
        variant      = _assign_treatment(install_date)
        op_hours     = round(rng.uniform(120, 260), 1)  # operating hours / month

        duration, event = _simulate_tte(region, crew, install_date, variant, ANALYSIS_END)

        records.append({
            "unit_id":               f"UNIT-{10000+i}",
            "install_date":          install_date.isoformat(),
            "region":                region,
            "install_crew":          crew,
            "design_variant":        variant,
            "operating_hours_month": op_hours,
            "time_to_event":         duration,
            "failure_event":         event,
        })

    df = pd.DataFrame(records)
    return df


def main():
    OUT_PATH.parent.mkdir(exist_ok=True)
    print(f"[data_generator] Generating {N_UNITS} unit panel...")
    df = generate_panel()
    df.to_csv(OUT_PATH, index=False)

    treat = df[df.design_variant == "B"]
    ctrl  = df[df.design_variant == "A"]
    print(f"[data_generator] Saved {len(df):,} units → {OUT_PATH}")
    print(f"  Variant A (control): {len(ctrl):,} units | failure rate: {ctrl.failure_event.mean()*100:.1f}%")
    print(f"  Variant B (treated): {len(treat):,} units | failure rate: {treat.failure_event.mean()*100:.1f}%")
    print(f"  True causal effect: HR = {TRUE_HAZARD_RATIO} (B reduces hazard by {(1-TRUE_HAZARD_RATIO)*100:.0f}%)")
    print(f"  Naive failure rate difference: {(treat.failure_event.mean() - ctrl.failure_event.mean())*100:+.1f}% (confounded by selection bias)")


if __name__ == "__main__":
    main()
