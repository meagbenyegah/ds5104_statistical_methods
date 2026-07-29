# Execution Plan: DS5104 Final Assessment

Source: `FINAL_EXAMS_STAT_METHODS_DATA_SCIENCE_26_IGIT.md`
Deadline: 29/07/2026 · α = 0.05 throughout unless stated otherwise.

This is a plan of attack, not filled-in answers — it maps each question to the
test/model to run, the code needed, and exactly what numbers and interpretive
statements have to appear in the final write-up. Run the code against the real
data, then paste results into a report (Word/PDF/Jupyter) organized by these
same headings.

## 0. Setup (do once)

- [ ] Locate/place `fitpulse_health_dataset.csv` in the project (not currently
      in this directory — needed before Part A can be computed).
- [ ] Confirm `seaborn` is installed for Part B (`sns.load_dataset("tips")`,
      N=244 — matches the brief, so no extra file needed).
- [ ] Python stack: `pandas`, `scipy.stats`, `statsmodels`, `sklearn`,
      `numpy`. R equivalents noted where useful (`shapiro.test`, `t.test`,
      `chisq.test`, `aov`, `glm`).
- [ ] Standard reporting template for every hypothesis test (reuse for all
      sub-questions): H0 / H1 → test statistic → df (if applicable) → p-value
      → decision rule (reject/fail to reject H0 vs α) → plain-language
      conclusion tied to the business/product context.

---

## Part A — FitPulse Health Dataset

### A1. Normality & Descriptive Metrics (15 pts)
- **Variables:** `daily_steps_avg`, `age`
- (a) H0: `daily_steps_avg` is drawn from a normal distribution. H1: it is not.
- (b) `scipy.stats.shapiro(df['daily_steps_avg'])`
  - Report W and p-value.
  - Decision: reject H0 if p < 0.05.
  - If non-normal → report **Median (IQR)**; if normal → **Mean (SD)**.
  - Justify from the W/p-value and ideally a QQ-plot/histogram sanity check.
- Repeat the same lens conceptually for `age` if the write-up expects both
  variables profiled (the point allocation focuses on `daily_steps_avg`, but
  compute `age` descriptives too — mean/SD or median/IQR per its own
  normality check).

### A2. Count Data Fundamentals (10 pts)
- **Variable:** `supplement_purchases` (non-negative integer counts)
- (a) Two theoretical reasons OLS fails for counts:
  1. OLS assumes a continuous, unbounded (−∞, ∞) response with homoscedastic
     normal errors; counts are discrete and bounded at 0, so predictions can
     go negative and residuals are heteroscedastic (variance grows with the
     mean).
  2. Count data is typically right-skewed with variance tied to the mean
     (Poisson-like), violating OLS's constant-variance assumption — leads to
     inefficient estimates and invalid standard errors/p-values.
- (b) Suitable distribution: **Poisson** (assumes mean = variance,
  independent events, fixed exposure period). Overdispersion (variance >
  mean) inflates standard errors if ignored under Poisson, causing
  understated p-values / overstated significance → invalidates the Poisson
  fit; remedy is Negative Binomial regression (extra dispersion parameter).

### A3. A/B Experimentation (15 pts)
- **Groups:** `ab_test_group` (Control vs Variant B) → outcome `daily_steps_avg`
- (a) H0: μ_control = μ_variantB. H1: μ_control ≠ μ_variantB (two-sided).
- (b) `scipy.stats.ttest_ind(control, variantB, equal_var=...)` — check
  Levene's test first to decide `equal_var`.
  - Report t, df, p-value, decision, and a ship/no-ship recommendation tied
    to effect size + business relevance (not just significance).
- (c) Type I (α): falsely conclude Variant B works → ship a feature with no
  real benefit, wasted engineering/marketing spend. Type II (β): miss a real
  improvement → forgo a genuine engagement gain, competitive cost.

### A4. Categorical Association (15 pts)
- **Variables:** `subscription_tier` (Free/Basic/Premium) × `churned_within_6mo`
- H0: subscription tier and churn are independent. H1: they are associated.
- `pd.crosstab(tier, churned)` → `scipy.stats.chi2_contingency(table)`
  - Report the observed table, the **expected frequencies table** (from the
    function's second return value), df = (rows−1)(cols−1), χ², p-value.
  - Decision + retention insight: identify which tier over/under-indexes on
    churn by comparing observed vs expected cells (standardized residuals =
    (obs−exp)/√exp are a nice add-on).

### A5. Multiple Linear Regression (20 pts)
- **Model:** `steps ~ age + sleep_quality`
- (a) `statsmodels.formula.api.ols('daily_steps_avg ~ age + sleep_quality', data=df).fit()`
  - Interpret β0 (expected steps at age=0, sleep_quality=0 — note
    extrapolation caveat), β1 (steps change per +1 year age, holding sleep
    quality constant), β2 (steps change per +1 unit sleep quality, holding
    age constant).
- (b) Partial t-tests: pull `model.tvalues` / `model.pvalues` per predictor,
  compare to α=0.05.
- (c) Multicollinearity = predictors linearly correlated with each other,
  inflating coefficient variance. VIF = 1/(1−R²_j) from regressing each
  predictor on the others (`statsmodels.stats.outliers_influence.variance_inflation_factor`).
  VIF > 5 (or 10) flags a problem. High VIF → inflated standard errors →
  unstable coefficients, wider CIs, unreliable individual t-tests even if
  overall model fit (R²) is fine.

### A6. Logistic Regression & Classification (25 pts)
- **Model:** `logit(churn) ~ age + Premium_tier`
- (a) Linear probability models can predict outside [0,1] and violate
  homoscedasticity/normality of errors for a binary outcome; the logit link
  constrains predictions to (0,1) and models log-odds linearly, which is the
  natural link for Bernoulli outcomes.
- (b) `sm.Logit(y, X).fit()` → OR = `np.exp(params)`. OR for age: multiplicative
  change in churn odds per +1 year, holding tier constant. OR for Premium:
  if OR < 1, Premium tier is associated with *lower* churn odds (protective);
  if OR > 1, higher churn odds. State explicitly whether OR_premium < 1.
- (c) Threshold predictions at 0.5 (or report curve), then:
  `sklearn.metrics.accuracy_score`, `precision_score`, `recall_score`,
  `roc_auc_score` — report all four with a sentence on what each implies for
  identifying at-risk (churning) users.

---

## Part B — Seaborn `tips` Dataset (N = 244)

### B1. Distributional Assessment (15 pts)
- **Variable:** `total_bill`
- Same recipe as A1: H0 normal / H1 not normal → `shapiro(tips.total_bill)`
  → W, p-value → decision → Mean(SD) vs Median(IQR) justification.

### B2. Party Size Modeling (10 pts)
- **Variable:** `size` (discrete counts, 1–6)
- (a) Same OLS-vs-count argument as A2, applied to `size`: bounded/discrete
  outcome, OLS can predict fractional or out-of-range party sizes, variance
  not constant across the mean.
- (b) Poisson assumes equidispersion (Var = Mean); when sample variance of
  `size` exceeds its mean (check `tips['size'].var()` vs `.mean()`),
  overdispersion → use Negative Binomial (or quasi-Poisson) instead.

### B3. Gender Spend Comparison (15 pts)
- **Groups:** `sex` × `total_bill`
- (a) H0: μ_male = μ_female. H1: μ_male ≠ μ_female.
  `ttest_ind(male_bills, female_bills, equal_var=...)` → t, df, p, decision.
- (b) Type I: change staffing/menu strategy based on a spurious gender
  difference, no real revenue gain, possible poor targeting. Type II: miss a
  genuine spend gap, forgo a valid segmentation/upsell opportunity.

### B4. ANOVA & Independence (15 pts)
- (a) One-way ANOVA: `total_bill ~ day` (Thur/Fri/Sat/Sun)
  - H0: all day means equal. H1: at least one differs.
  - `scipy.stats.f_oneway(*groups)` or `statsmodels` `ols(...).fit()` +
    `anova_lm` for a full table (SS, df, F, p).
  - If significant → recommend post-hoc **Tukey HSD**
    (`statsmodels.stats.multicomp.pairwise_tukeyhsd`) to find which day
    pairs differ.
- (b) Chi-square independence: `sex` × `smoker`
  - `pd.crosstab` → `chi2_contingency` → df, χ², p-value, conclusion on
    whether smoking status is associated with sex.

### B5. Tip Prediction Regression (20 pts)
- **Model:** `tip ~ total_bill + size`
- (a) `ols('tip ~ total_bill + size', data=tips).fit()` — interpret β1 (extra
  tip per +$1 bill, holding party size constant) and β2 (extra tip per
  +1 party member, holding bill constant).
- (b) Partial t-tests on both predictors (t-values/p-values from summary).
- (c) VIF for `total_bill` and `size` — note they're plausibly correlated
  (bigger parties → bigger bills), discuss whether VIF crosses the 5/10
  threshold and what that implies for trusting individual coefficients.

### B6. High-Tip Binary Logistic Model (25 pts)
- **Outcome:** `high_tip = 1 if tip >= 3.00 else 0`
- **Model:** `logit(high_tip) ~ total_bill + smoker`
- (a) Same linear-probability-vs-logit argument as A6(a), specific to
  bounding predicted probabilities in [0,1] for this binary target.
- (b) `sm.Logit(y, X).fit().summary()` → report β, SE, z, p for each
  predictor (note: logistic regression reports z-scores, not t-scores).
- (c) OR = exp(β) for `total_bill` (odds multiplier per +$1 bill) and
  `smoker` (odds multiplier for smoker vs non-smoker) → translate into a
  staffing insight (e.g., "smoker tables are X% more/less likely to tip
  ≥$3, suggesting...").

---

## Deliverable checklist

- [ ] One consolidated report (notebook or doc) with Part A (Q1–6) and
      Part B (Q1–6) in the same order as the brief.
- [ ] Every hypothesis test states H0/H1 explicitly before results.
- [ ] Every test reports statistic + df (where applicable) + p-value +
      explicit reject/fail-to-reject decision.
- [ ] Every numeric result gets one line of plain-language interpretation
      tied to the FitPulse/restaurant business context — not just the stats.
- [ ] Regression sections include coefficient tables + VIF tables, not just
      prose.
- [ ] Fill in INDEX NUMBER on the submission.
