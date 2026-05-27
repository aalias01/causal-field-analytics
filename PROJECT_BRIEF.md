# Project Brief — Causal Field Analytics

| Priority Score | Tier | Recommended Ship Slot | Effort |
|----------------|------|----------------------|--------|
| **3.95** | **P2** | **Order #9** *(after CMAPSS · Retail Returns · RAG · HVAC · Supply Chain · Energy Demand · Industrial Failure Classification · Maintenance NLP)* | 26–32 hrs across 5 sessions *(expanded April 2026 — added survival analysis track from Apr 2026 classmate method-depth pass)* |

**Score breakdown** — ED 4 · DIFF 4.5 · SC 4 · DSS 5 · BV 3 · EE 3 *(DIFF nudged 4 → 4.5 after the April 2026 classmate method-depth pass added a survival-analysis track — combining causal inference WITH survival analysis on industrial reliability data is a near-unique cross at the entry-DS level)*
**Working title:** *Causal Field Analytics — Estimating the Impact of Engineering Design Changes on Field-Failure Rates from Observational Data*
**Lane:** Cross-lane (causal inference applies equally to retail/ops and industrial DS)
**Target companies:** Costco (promo-vs-control experimentation), Amazon / Airbnb / Uber / Netflix (experimentation teams), Boeing (field reliability), GE Vernova (fleet analytics), Honeywell (building-system reliability)

**Conditions to re-rank:**
- **BV = 3** because clean real observational industrial datasets are rare in the public domain. If a strong dataset surfaces (e.g. a public automotive recall panel, an energy-grid intervention dataset, or a well-documented manufacturing design-change panel), promote BV to 4–5 and reconsider for P1.
- **DSS = 5** — Alvin's 7 years of observational field-failure judgment (Daikin sustaining engineering + Rheem product releases) is the *actual* moat. Most DS candidates can run `DoWhy.fit()` but can't tell you which confounders matter in a real failure-rate panel.
- **DATA 557 Applied Statistics & Experimental Design** (UW MSDS coursework) provides the statistical foundation — the project is the *applied* complement to the coursework.
- If applying to a causal-inference-heavy team (e.g. Amazon Core Science, Airbnb Experimentation, Netflix Experimentation & Causal Inference): **promote tactically** to demonstrate directly relevant skill.

---

## Why This Brief Exists (Origin)

Added April 2026 as part of the **peer-benchmark / strategic pass** (see `portfolio_pipeline.md` → *Strategic Pass — April 2026 ("Market-Signal Audit")*). A UW MSDS classmate's resume explicitly features causal inference ("causal inference and correlation to identify the main revenue drivers" at Woods Coffee), and the 2026 DS hiring market — especially at retail/ops employers (Costco, Amazon, Airbnb, Uber, Netflix) — treats causal inference / experimental design as a top-3 skill alongside ranking/recsys and LLM fluency.

The prior portfolio had only an A/B-test stub inside the Retail Returns project. This brief gives causal inference its own dedicated artifact and leans directly into Alvin's strongest moat — 7 years of observational field-data judgment no bootcamp graduate can replicate.

---

## Problem Statement

Engineering organizations constantly make design changes — a supplier swap, a firmware update, a material substitution, a refrigerant-regulation-driven redesign — and then ask *"did this change actually improve field reliability?"*

The naive answer (*"compare failure rates before and after"*) is wrong because:

- **Seasonality confounds the trend** — HVAC equipment fails more in July than in January, regardless of design
- **Install-crew and region confound the signal** — a design-change rollout that happened in the Southeast first will look better or worse than one that rolled out in the Midwest first
- **Selection bias in who receives the change** — newer units get the redesign; newer units have baseline-lower failure rates anyway
- **Reporting lag** — units installed six months ago haven't had enough time to fail yet

Given a panel of equipment units with: install date, region, install crew, design variant (treatment vs. control), and observed failures over time — **estimate the causal impact of the design variant on failure rate while controlling for these confounders.**

Deliverables: a Quarto analytical report that a reliability-engineering stakeholder would find actually useful, plus a companion notebook showing the statistical machinery honestly (what each method assumes, when it breaks).

---

## Why This Project for Alvin

- **Unique lived experience.** Alvin *did this work* at Daikin (sustaining engineering, field-failure root cause, supplier-cost optimization) and Rheem (2023 DoE transition, 2025 refrigerant transition). He's lived the confounders — regional climate, install crew, seasonality, reporting lag. His statistical intuition here is *real*, not read from a textbook.
- **Closes the peer-benchmark gap.** Directly addresses the "causal inference" axis surfaced in the April 2026 peer-benchmark pass.
- **Leverages UW coursework.** DATA 557 Applied Statistics & Experimental Design is already on the transcript — this is the applied project that turns the coursework into a resume bullet.
- **Cross-lane.** Unlike HVAC Health Scoring (industrial-only) or Retail Returns (retail-only), causal inference applies equally to Costco promo analytics, Amazon experimentation, Boeing field reliability, and GE Vernova fleet analytics.
- **Differentiator.** The median DS candidate can run an A/B test. Very few can frame *why* a naive before/after comparison lies to you, which methods to reach for, and what their identifying assumptions demand from the data. This project produces that exact artifact.

---

## Dataset Strategy

**Order of preference: Real public → Real public + documented augmentation → Carefully-flagged simulation.**

### Tier 1 — Real public datasets (try first)

- **SECOM semiconductor manufacturing (UCI)** — 1,567 examples, 590 sensor features, binary pass/fail. Not explicitly causal but contains process-parameter variants that can be framed as treatment; requires creative problem framing.
- **NASA PCoE bearing / battery degradation** — includes operating-condition variants that can be framed as treatment vs. control panels.
- **CDC MMWR or FDA recall panels** — public, structured, with date/region confounders that match the industrial framing. Promotes reusability for healthcare-adjacent roles.
- **EIA electricity grid intervention data** — regional policy changes treated as natural experiments; strong fit if Energy Demand Forecasting ships first and shares vocabulary.

**If one of these fits cleanly:** promote BV to 4 and reconsider for P1.

### Tier 2 — Real data + documented simulation layer

Use a real equipment-panel dataset as the anchor (install dates, regions, unit IDs) and **layer on a clearly-flagged simulated treatment-and-outcome structure**. The simulated part is *the causal structure*, not the panel itself. This is a well-accepted practice in causal inference teaching (see `DoWhy` tutorials) as long as the simulation is transparent.

### Tier 3 — Fully simulated panel (last resort, flagged)

Generate a synthetic field-panel dataset using Alvin's domain knowledge:
- ~5,000 equipment units, 24-month observation window
- Features: install date, region (4 levels), install crew (3 levels), unit age, operating-hours per month
- Treatment: design variant A vs. design variant B, assigned non-randomly (newer units more likely to get B — a realistic selection bias)
- True causal effect: variant B reduces failure hazard by 15% (this is the ground truth the model must recover)
- Observed failure events + right-censoring

**If synthetic-only:** explicitly state in the README that the panel is simulated with known ground-truth causal effect, and position the project as *"a causal-inference methodology showcase where the ground truth is known, used to validate that each method recovers the true effect under varying confounder strength."* This is actually a *stronger* framing for a learning-signal project than using a real dataset where ground truth is unknown.

---

## Methodology — Four Causal Approaches Compared

The value of this project is showing that **you pick the right method for the question**, not that you run all four blindly.

### Method 1 — Naive before/after comparison (as the straw-man)
- Plot failure rate before vs. after design change
- Deliberately wrong; included so the stakeholder can *see* what the confounders cost
- Hook line for the report: *"This naive view says the change made things worse. It didn't. Here's what the naive view is missing."*

### Method 2 — Propensity Score Matching
- Estimate `P(treatment | confounders)` via logistic regression or gradient boosting
- Match treated units to similar-propensity control units
- Estimate Average Treatment Effect on the Treated (ATT) on matched sample
- Libraries: `causalinference`, `dowhy`, or hand-implemented with `scikit-learn`
- Discuss the positivity and unconfoundedness assumptions honestly

### Method 3 — Difference-in-Differences (DiD)
- Applicable if treatment rolls out *at a specific time* to *a specific subset of units*
- Compute `(treated_after − treated_before) − (control_after − control_before)`
- Fit via OLS with unit and time fixed effects (`statsmodels`)
- Parallel-trends assumption — *test it*, don't assume it; plot pre-treatment trajectories

### Method 4 — Synthetic Control
- When there's one treated group and many control groups (e.g. one region got the variant, others didn't)
- Construct a weighted average of control regions that best matches the treated region's *pre-treatment* trajectory
- Compare post-treatment outcomes
- Library: `pysyncon` or hand-implemented
- Good framing for the report: *"The 'Southeast rollout' case study — what would the Southeast's failure rate have been if we'd never made the change?"*

### Variance Reduction — CUPED *(stretch)*
- Controlled-experiment Using Pre-Experiment Data — a Microsoft technique now standard at Amazon, Airbnb, Netflix
- Reduces variance of treatment-effect estimates by adjusting outcomes for a pre-experiment covariate
- Even if the core project uses observational methods, a CUPED appendix demonstrates A/B-testing fluency at the level real experimentation teams operate at

### Method 5 — Survival Analysis *(new — Apr 2026 classmate method-depth pass)*

**Why this fits exactly here.** A field-failure panel is *literally* a survival dataset: install date = enrollment, failure event = death, units that haven't failed yet = right-censored. Reliability engineering's core vocabulary (MTBF, hazard rate, Weibull failure curves, B10 life) is just survival analysis with industrial terminology — the same math, different audience. The other four causal methods estimate **how big** the design-change effect is on outcome rates; survival analysis estimates **how the effect changes the failure-time *distribution* itself**. Reliability teams care more about the latter (warranty exposure is a *time-integrated* quantity).

- **Goal:** Estimate the causal effect of the design variant on the *time-to-failure distribution* — not just the average failure rate
- **Algorithms:**
  - **Kaplan-Meier estimator** — non-parametric survival curves for treatment vs. control, with log-rank test for significance
  - **Weibull / accelerated failure time (AFT) models** — parametric reliability-engineering standard; produces interpretable shape and scale parameters for both groups (`lifelines` library)
  - **Cox proportional hazards regression** — semi-parametric; controls for the same confounders the propensity / DiD methods use; outputs hazard ratio of treatment with CIs
  - **DeepSurv (stretch)** — neural-network extension of Cox PH; relaxes the proportional-hazards assumption; demonstrates modern survival ML
- **Library:** `lifelines` (primary, well-documented; the de-facto Python survival library), `scikit-survival` (gradient boosting for survival), `pycox` (DeepSurv stretch)
- **Diagnostics that matter:**
  - Schoenfeld residuals for the proportional-hazards assumption (Cox PH is invalid if violated)
  - Goodness-of-fit for Weibull (Q-Q plot of empirical vs. theoretical quantiles)
  - Log-rank test power calculation given panel size
- **Industrial framing for the report:** *"What's the B10 life under the old design vs. the new one — i.e., the time at which 10% of units have failed? That's the warranty-exposure number a finance team can act on."* This sentence makes the project *unmissably* legible to any reliability or warranty team and *novel* to a causal-inference-only audience.

**Why this is high-leverage specifically for Alvin:**

- **Domain × technique cross.** Healthcare-DS and biostatistics-heavy profiles use survival analysis on patients. Industrial-DS folks rarely apply it formally to field data, even though the underlying math is the *foundation* of reliability engineering. Alvin can authentically write code that uses biostatistics-grade survival methods AND speak to the reliability-engineering audience that owns the data — almost no one in the entry-DS market can credibly do both.
- **Career arc.** This is the technique that bridges Alvin's industrial past (where MTBF is a daily-use term) to a modern statistical methodology (where Cox PH and DeepSurv live). On a resume, "Cox proportional hazards regression on industrial field-failure panels with hazard-ratio confidence intervals" reads as fluent in *both* dialects.
- **Tied to UW coursework.** DATA 557 covers regression including time-to-event; this lets Alvin extend the coursework one principled step further.

---

## Tech Stack

| Layer | Tool | Justification |
|-------|------|---------------|
| Notebook environment | Jupyter (conda) | Standard exploratory loop |
| Data wrangling | Pandas, NumPy | Panel manipulation |
| Statistical modeling | **statsmodels** | OLS with fixed effects for DiD; regression diagnostics |
| Causal-inference framework | **DoWhy** (primary) + **EconML** (supplement) | DoWhy for explicit identification → estimation → refutation workflow; EconML for modern estimators |
| Propensity scoring | scikit-learn `LogisticRegression` + `GradientBoostingClassifier` | |
| Matching | `causalinference` library | |
| Synthetic control | `pysyncon` or hand-implemented | |
| CUPED *(stretch)* | statsmodels + custom adjustment | |
| **Survival analysis** *(new)* | **`lifelines`** (Kaplan-Meier, Cox PH, Weibull AFT) + **`scikit-survival`** (gradient boosting survival) | `lifelines` is the Python survival standard; `scikit-survival` integrates with sklearn API |
| **Survival ML stretch** | **`pycox`** (DeepSurv, DeepHit) | Modern neural-network survival; demonstrates beyond-Cox depth |
| Visualization | Matplotlib, Seaborn, Plotly | Trajectory plots; effect-size forest plots; parallel-trends diagnostics |
| Reporting | **Quarto** → rendered HTML on Vercel | Same pattern as Supply Chain + Energy Forecasting; no API needed |
| Version control | git + GitHub repo `aalias01/causal-field-analytics` | |
| Environment | conda (`environment.yml`) | |

---

## Deliverables

1. `notebooks/01_data_understanding.ipynb` — panel structure, missingness, treatment assignment patterns, confounder exploration
2. `notebooks/02_naive_straw_man.ipynb` — before/after + simple group-comparison; document what the naive view misses
3. `notebooks/03_propensity_score_matching.ipynb` — propensity model, matching, balance diagnostics, ATT estimation
4. `notebooks/04_difference_in_differences.ipynb` — DiD with unit + time fixed effects, parallel-trends diagnostics
5. `notebooks/05_synthetic_control.ipynb` — weighted-control construction, post-treatment comparison, placebo tests
6. `notebooks/06_cuped_appendix.ipynb` *(stretch)* — variance reduction on A/B-test data
7. `notebooks/07_survival_analysis.ipynb` *(new)* — Kaplan-Meier curves with log-rank test, Weibull AFT fits, Cox PH with confounders, hazard-ratio forest plot, B10-life comparison table
8. `notebooks/08_method_comparison.ipynb` — side-by-side effect estimates across all methods (causal + survival), CIs, identifying-assumption comparison
9. `src/estimators.py` — reusable functions for each method (incl. Cox PH and Weibull fitting)
10. `src/diagnostics.py` — propensity balance plots, parallel-trends tests, Schoenfeld residual diagnostics, log-rank tests
11. `report/causal_field_analytics.qmd` — Quarto report, rendered to static HTML on Vercel
12. `README.md` — recruiter-facing, clear statement of ground truth recovered vs. naive bias, with a *separate paragraph for reliability engineers* on the B10-life and hazard-ratio findings
13. (optional) GitHub Pages render of the Quarto report for link-sharing

---

## Project Phases

### Phase 1 — Data + Setup + Causal DAG (4–5 hrs)
- [ ] Pick dataset tier (aim for Tier 1; fall back to Tier 2/3 with clear flagging)
- [ ] Explore panel: unit counts, time coverage, treatment assignment mechanism, confounder distributions
- [ ] Draw the causal DAG explicitly (DoWhy supports this): `install_region → treatment`, `install_date → treatment`, `install_date → outcome`, `treatment → outcome`
- [ ] Document identifying assumptions for each method
- [ ] Output: `notebooks/01_data_understanding.ipynb` + causal DAG image for the report

### Phase 2 — Naive Baseline + Propensity Score Matching (4–5 hrs)
- [ ] Naive before/after: visually striking plot of *wrong* conclusion
- [ ] Fit propensity model; inspect overlap (common support); trim where needed
- [ ] Nearest-neighbor matching; check covariate balance pre/post
- [ ] ATT estimation with bootstrap CI
- [ ] Sensitivity analysis — how large must unobserved confounding be to overturn the result? (Rosenbaum bounds or E-value)
- [ ] Output: `notebooks/02_naive_straw_man.ipynb` + `notebooks/03_propensity_score_matching.ipynb`

### Phase 3 — Difference-in-Differences (3–4 hrs)
- [ ] Parallel-trends check on pre-treatment window; statistical test + visual
- [ ] OLS with unit + time fixed effects (`statsmodels`)
- [ ] Placebo test: pretend the treatment happened earlier; effect should be zero
- [ ] Output: `notebooks/04_difference_in_differences.ipynb`

### Phase 4 — Synthetic Control + Method Comparison (4–5 hrs)
- [ ] Construct synthetic control weights over pre-treatment trajectory
- [ ] Compute gap between actual treated and synthetic control post-treatment
- [ ] Placebo/permutation inference (apply to untreated units to build null distribution)
- [ ] Side-by-side table: all four methods with effect estimates, CIs, and each method's identifying assumption
- [ ] Output: `notebooks/05_synthetic_control.ipynb` + `notebooks/07_method_comparison.ipynb`

### Phase 5 — Survival Analysis Track *(new, 5–6 hrs)*
- [ ] Convert panel into survival-format dataframe: `(unit_id, duration, event_observed, treatment, covariates)` with right-censoring for units that haven't failed by end-of-window
- [ ] Kaplan-Meier survival curves for treatment vs. control with log-rank test (`lifelines.KaplanMeierFitter`, `lifelines.statistics.logrank_test`)
- [ ] Weibull AFT fit to each group; report shape and scale parameters; check goodness-of-fit via Q-Q plot
- [ ] Cox proportional hazards regression with all confounders (region, install crew, install date, age) — report hazard ratio of treatment with 95% CI
- [ ] Schoenfeld residual diagnostic for the proportional-hazards assumption — flag honestly if violated
- [ ] B10-life comparison table: *"under the old design, 10% of units fail by month X; under the new design, by month Y"* — the reliability-engineering bottom line
- [ ] Hazard-ratio forest plot integrating into the method-comparison panel from Phase 4
- [ ] Stretch: DeepSurv via `pycox` for non-proportional-hazards handling
- [ ] Output: `notebooks/07_survival_analysis.ipynb`

### Phase 6 — CUPED Appendix *(stretch, 2–3 hrs)*
- [ ] Build a side A/B-test scenario; apply CUPED with pre-experiment covariate
- [ ] Show variance reduction vs. standard t-test
- [ ] Output: `notebooks/06_cuped_appendix.ipynb`

### Phase 7 — Quarto Report + Ship (3–4 hrs)
- [ ] Write the report from *two* stakeholder perspectives:
  - For the experimentation / causal team: *what's the question, what would the naive answer say, what's the right answer, how confident are we*
  - For the reliability / warranty team: *what's the hazard-ratio comparison, what's the B10-life delta, what does this mean for warranty exposure*
- [ ] Embed effect-size forest plot (all five methods side by side: naive, PSM, DiD, synthetic control, Cox PH)
- [ ] Embed Kaplan-Meier survival curves overlay
- [ ] Render Quarto HTML; deploy to Vercel as static site
- [ ] Update README with two headlines:
  - *"Causal: naive view estimated +X% failure-rate impact (wrong direction); causal methods converged on −Y% ± Z (95% CI)"*
  - *"Reliability: B10-life increased from M1 months under old design to M2 months under new design (Cox HR = 0.YZ, 95% CI [a, b])"*
- [ ] Log in `shipped.qmd`
- [ ] Propagate resume bullet to `resume_facts.md`

**Total: 23–28 hrs base + 2–3 hrs CUPED stretch + 1–2 hrs DeepSurv stretch = 26–32 hrs across 5 sessions.**

---

## Interview Talking Points

1. **Why this project at all?** *"Most DS candidates can run an A/B test. Very few can explain why a naive before/after comparison lies to you in real observational data — which is almost all the data any real company has. This project shows I know the difference between correlation and causation in a way that a field-reliability team at Boeing or an experimentation team at Amazon would both recognize."*
2. **Why four methods?** *"Each method has a different identifying assumption. Propensity matching assumes no unmeasured confounders. DiD assumes parallel trends. Synthetic control requires a long enough pre-treatment window. The right answer is to show the estimates converge — or to honestly report where they don't, and why. Stakeholders trust that kind of honesty more than a single flashy number."*
3. **Why it connects to the engineering background.** *"At Daikin I did field-failure root cause for four years. I've seen teams declare victory on a design change that was really just a seasonal dip, and I've seen good changes get blamed for problems they didn't cause. This is the toolkit I wish we'd had — and it's what I'd build first if a reliability team brought me in on day one."*
4. **Why it connects to retail/ops.** *"The exact same methodology answers 'did the new promotion actually lift revenue?' or 'did the new returns policy actually reduce abuse?' — the questions Costco and Amazon ask every week. Propensity score matching on customer features, DiD on region rollouts, CUPED for variance reduction — it's the retail experimentation stack."*
5. **The ground-truth move.** *"If the panel is simulated with known true effect, I actively use that — I run each method, see how close it gets to the truth, and show where each one breaks. That's a teaching-through-demonstration move that a real dataset can't give you, because a real dataset never tells you what the truth was."*
6. **Why survival analysis specifically.** *"A field-failure panel is literally a survival dataset — install date is enrollment, failure event is the death, units that haven't failed yet are right-censored. Reliability engineering's whole vocabulary — MTBF, hazard rate, Weibull failure curves, B10 life — is just biostatistics survival analysis with industrial vocabulary. I can write Cox proportional hazards regression with the same fluency I'd compute MTBF, and I can talk to a reliability engineer or a biostatistician with the same toolkit. That cross is rare at the entry-DS level — most candidates have one dialect or the other."*

---

## Success Criteria

- [ ] Quarto report rendered on Vercel with a clear, stakeholder-readable headline
- [ ] GitHub repo public with all 7 notebooks reproducible from `environment.yml`
- [ ] Causal DAG explicitly drawn and explained in the report
- [ ] At least 3 of the 4 methods completed (propensity matching, DiD, and one of synthetic control or method comparison)
- [ ] Effect-size forest plot showing all methods side by side with CIs
- [ ] Sensitivity analysis (Rosenbaum bounds or E-value) documented
- [ ] Resume bullet drafted (see below)

### Resume bullets (drafts)

**For causal-inference-heavy roles (retail/ops experimentation, field reliability)** — *"Built causal-inference analytical report estimating the impact of an engineering design change on field-failure rates from observational panel data: propensity-score matching, difference-in-differences with unit and time fixed effects, synthetic control, Cox proportional hazards survival regression, and CUPED variance reduction — with sensitivity analysis (Rosenbaum bounds), Schoenfeld residual diagnostics, and method-comparison forest plot. Deployed as Quarto report on Vercel."*

**For reliability / warranty / industrial roles** *(new variant)* — *"Applied biostatistics-grade survival analysis (Kaplan-Meier with log-rank test, Weibull AFT, Cox proportional hazards regression with confounder adjustment) to industrial field-failure panels, producing hazard-ratio confidence intervals and B10-life comparisons for warranty-exposure decisioning."*

**For general DS roles (lighter version)** — *"Applied causal inference (propensity matching, DiD, synthetic control, Cox PH) to estimate design-change impact from observational equipment-panel data; surfaced naive-comparison bias of X% and reported causal effect of Y% ± CI with sensitivity analysis."*

---

## Reading / Reference (for the build)

- *Causal Inference: The Mixtape* — Scott Cunningham (free online)
- *DoWhy* documentation — the identification → estimation → refutation workflow
- *Microsoft CUPED paper* — Deng et al. 2013 "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data"
- UW DATA 557 course materials — leverage existing coursework
- *lifelines documentation* — the survival-analysis Python standard; the "Survival Regression" tutorial is a 90-min onramp
- *Survival Analysis: A Self-Learning Text* — Kleinbaum & Klein (the standard textbook; chapter on Cox PH is the one that matters)
- *Reliability Engineering and Risk Analysis* — Modarres et al. — for the industrial-engineering vocabulary translation (MTBF ↔ mean of survival distribution; B10 life ↔ 10th percentile of failure-time distribution)
- *DeepSurv paper* — Katzman et al. 2018 (only if pursuing the stretch)

---

*Brief created: April 2026 (strategic pass — market-signal audit) | Updated April 2026 (classmate method-depth pass — Survival Analysis track added) | May 2026 (activated; slot confirmed #9) | Tier P2 · Ship slot #9 · Cross-lane: industrial reliability + retail experimentation · Promote to P1 if real causal dataset surfaces or survival-analysis-heavy JD appears*
