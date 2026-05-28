# Portfolio Readiness Checklist

This repo is much closer to portfolio-ready now: the data generator runs, the headline metrics are reproducible, figures are exported, the README has real numbers, and the Quarto report no longer depends on placeholder text.

## What Was Completed Locally

- Created the `causal-field` conda environment from `environment.yml`.
- Updated the synthetic rollout so it matches the intended story: variant B is genuinely protective, but it is rolled out first in the Southeast, a higher-hazard region.
- Added `scripts/portfolio_analysis.py` as the reproducible analysis entry point.
- Generated:
  - `data/field_panel.csv` (gitignored)
  - `figures/naive_confounders.png`
  - `figures/propensity_overlap.png`
  - `figures/parallel_trends.png`
  - `figures/synthetic_control.png`
  - `figures/kaplan_meier.png`
  - `figures/method_comparison_forest.png`
  - `report/portfolio_metrics.json`
- Updated `README.md` with actual results.
- Rewrote `report/causal_field_analytics.qmd` with the real findings and honest diagnostics.
- Synchronized `requirements.txt` with the declared environment.

## Current Headline Results

| Method | Result | Portfolio Interpretation |
|---|---:|---|
| Naive comparison | +7.11% relative failure increase | Wrong direction because rollout is confounded |
| Propensity score matching | -4.97 pp, 95% CI [-7.52, -2.27] | Main causal failure-rate estimate |
| Difference-in-differences | -3.70 pp, 95% CI [-9.48, 2.08] | Directionally consistent but imprecise |
| Synthetic control | +10.50 pp gap | Reject due to poor pre-treatment fit |
| Cox PH | HR = 0.812, 95% CI [0.718, 0.918] | Main reliability estimate |
| Cox-adjusted B10 | 6 months to 8 months | +2 month warranty-window improvement |

## Reproduce The Analysis

From the repo root:

```bash
conda activate causal-field
python scripts/portfolio_analysis.py
quarto render report/causal_field_analytics.qmd
```

If your shell does not expose `python` after activation:

```bash
conda run -n causal-field python scripts/portfolio_analysis.py
quarto render report/causal_field_analytics.qmd
```

Expected rendered output:

```text
report/causal_field_analytics.html
```

## Before Pushing To GitHub

1. Inspect the rendered report locally:

```bash
open report/causal_field_analytics.html
```

2. Confirm no generated private/local files are staged:

```bash
git status --short
```

3. Recommended files to stage:

```bash
git add README.md PORTFOLIO_READINESS.md requirements.txt .gitignore src/data_generator.py scripts/portfolio_analysis.py report/causal_field_analytics.qmd report/portfolio_metrics.json figures/*.png
```

4. Commit:

```bash
git commit -m "Make causal field analytics portfolio-ready"
```

5. Push:

```bash
git push origin main
```

## Deploy The Report

### Option A: GitHub Pages

1. Render the report:

```bash
quarto render report/causal_field_analytics.qmd
```

2. In GitHub, go to repository settings, then Pages.
3. Set the source to the branch and folder that contains the rendered HTML.
4. After the Pages URL is live, replace the README live-report line with the final URL.

### Option B: Vercel Static Deploy

1. Import the GitHub repository into Vercel.
2. Use the repo root as the project root.
3. Set the output/static directory to `report` if Vercel asks.
4. Make sure `report/causal_field_analytics.html` exists before deploy.
5. After deployment, update the README live-report URL.

## Remaining Portfolio Polish

- Add one screenshot of the rendered report to the README after deployment.
- Decide whether to keep the notebook stubs or fill notebooks 01, 02, 03, 07, and 08 from `scripts/portfolio_analysis.py`.
- Add a short resume bullet:

```text
Built causal-inference reliability analysis on a synthetic 5K-unit field-failure panel with known ground truth; showed naive rollout analysis reversed the conclusion, while PSM and Cox PH recovered the protective design effect (Cox HR 0.812 vs true 0.85) and improved Cox-adjusted B10 life from 6 to 8 months.
```

- Make the GitHub repo public only after the rendered report link works and the README link has been updated.
