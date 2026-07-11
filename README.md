# Causal Field Analytics

I estimate how an engineering design change affects field-failure rates when deployment was not randomized, then translate the result into hazard ratios and B10 life. Five methods run against a synthetic panel with known ground truth, so I can measure where each method succeeds or breaks.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![DoWhy](https://img.shields.io/badge/DoWhy-0.11-purple)](https://www.pywhy.org/dowhy/)
[![lifelines](https://img.shields.io/badge/lifelines-0.28-teal)](https://lifelines.readthedocs.io/)
[![Quarto](https://img.shields.io/badge/Quarto-report-orange)](https://quarto.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

**[Live report](https://aalias01.github.io/causal-field-analytics/report/causal_field_analytics.html)**

![Method comparison forest plot: naive and synthetic-control estimates miss; PSM and Cox PH recover the true effect](figures/method_comparison_forest.png)

## The problem

Engineering organizations make design changes (a supplier swap, a firmware update, a refrigerant-driven redesign) and then ask whether the change improved field reliability. The naive answer, comparing failure rates before vs after, is wrong here by construction:

```
Naive before/after: "the design change made failure rates WORSE (+7.11%)"

What the naive view misses:
  seasonality        HVAC fails more in July regardless of design
  selection bias     newer units got the redesign AND have lower baseline failure rates
  regional rollout   Southeast first, and its climate differs
  reporting lag      recently installed units haven't had time to fail yet

Causal methods controlling for these: "the change REDUCED failure probability
by 4.97 percentage points (95% CI: [-7.52, -2.27])"

Survival framing for reliability teams: "Cox-adjusted B10 life moved from
6 to 8 months (Cox HR = 0.812)"
```

I spent 7 years doing observational field-failure analysis at Daikin and Rheem; this project is that judgment written down as statistics.

## Results

The generator plants a true effect: variant B reduces failure hazard by 15%. Each method either recovers it or fails in a diagnosable way.

| Method | Effect estimate | 95% CI | Verdict |
|--------|----------------|--------|---------|
| Naive before/after | +7.11% relative failure increase | n/a | Confounded; wrong direction (the straw man) |
| Propensity Score Matching | -4.97 pp | [-7.52, -2.27] | Main causal failure-rate estimate |
| Difference-in-Differences | -3.70 pp | [-9.48, 2.08] | Directionally consistent, underpowered |
| Synthetic Control | +10.50 pp gap | n/a | Rejected: poor pre-treatment fit |
| Cox PH hazard ratio | 0.812 | [0.718, 0.918] | Main reliability estimate |
| Cox-adjusted B10 life, old design | 6 months | n/a | 10% cumulative failure, reference covariates |
| Cox-adjusted B10 life, new design | 8 months | n/a | A 2-month warranty-window improvement |

The point of running five methods is seeing where estimates converge and reporting where they don't. Synthetic control fails here and the report says why (single-region rollout with poor donor fit) instead of hiding it.

| Method | Identifying assumption |
|--------|------------------------|
| PSM | No unmeasured confounders |
| DiD | Parallel pre-treatment trends |
| Synthetic control | Pre-treatment trajectory fit |
| Cox PH | Proportional hazards (testable, tested) |

## The data

A synthetic field panel from `src/data_generator.py`: ~5,000 units, 24-month observation window, right-censoring, with region (4), install crew (3), design variant (treatment), and time-to-event fields.

The rollout is intentionally non-random: variant B enters the Southeast first, and the Southeast has a higher baseline hazard. That's what makes the naive estimate point the wrong way, which is the exact stakeholder problem this project exists to demonstrate.

Why synthetic with known truth: a real dataset never tells you what the right answer was. A planted 15% hazard reduction lets me measure how close each method gets and show exactly how each one breaks under confounding. A real dataset can't do that.

## Tech stack

Python 3.11, DoWhy 0.11 (identification, estimation, refutation), causalinference (PSM), statsmodels (DiD with fixed effects), pysyncon (synthetic control), lifelines 0.28 (Kaplan-Meier, Cox PH, Weibull AFT), EconML (DML, DR-Learner), Quarto report published to GitHub Pages.

## Run it locally

```bash
git clone https://github.com/aalias01/causal-field-analytics
cd causal-field-analytics

conda env create -f environment.yml
conda activate causal-field
python -m ipykernel install --user --name causal-field

python src/data_generator.py            # generate the panel, runs in seconds
python scripts/portfolio_analysis.py    # reproduce metrics and figures
quarto render report/causal_field_analytics.qmd
```

If `python` isn't on your shell path: `conda run -n causal-field python scripts/portfolio_analysis.py`.

## Limitations

- All data is synthetic and disclosed as such; the project demonstrates method selection and diagnosis, not a field result.
- PSM's no-unmeasured-confounders assumption holds here because the generator's confounders are all observed. Real field data offers no such guarantee, which is why the refutation step exists.
- DiD is underpowered at this panel size; its CI crosses zero.

Built by [Alvin Alias](https://github.com/aalias01), MS Data Science, University of Washington. 7 years of field-failure analytics at Daikin and Rheem.
