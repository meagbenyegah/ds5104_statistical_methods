# DS5104: Statistical Methods for Data Science — Final Assessment Answers

INDEX NUMBER: _______________

All tests conducted at α = 0.05 unless stated otherwise. Computations reproduced
in [`src/run_analysis.py`](../src/run_analysis.py) using `pandas`, `scipy`,
`statsmodels`, and `scikit-learn` against `data/fitpulse_health_dataset.csv`
(N = 1000) and `seaborn.load_dataset("tips")` (N = 244).

---

## Part A: FitPulse Health Dataset Analysis

### A1. Normality & Descriptive Metrics (15 pts)

**(a) Hypotheses (4 pts)**
H0: `daily_steps_avg` is drawn from a normally distributed population.
H1: `daily_steps_avg` is not drawn from a normally distributed population.

**(b) Shapiro-Wilk test (11 pts)**

1. **Test statistic:** W = 0.9844, p = 7.22 × 10⁻⁹
2. **Decision:** p < 0.05 → **reject H0**. `daily_steps_avg` significantly
   deviates from normality (skew = 0.378, kurtosis = −0.145 — mild right
   skew; with N = 1000 the test is highly sensitive to small deviations, but
   the result stands at the stated α).
3. **Metric choice:** Report **Median (IQR)**: Median = 3184.00,
   IQR = 2374.75 – 4163.25 (width 1788.50).
4. **Justification:** Since normality is formally rejected, the median/IQR
   are robust to the residual skew/outliers and don't assume a symmetric
   distribution the way Mean (SD) does; Mean (SD) = 3267.01 (1275.52) is
   reported for reference but is the secondary metric here.

*Supporting check on `age`:* W = 0.9527, p = 2.06 × 10⁻¹⁷ → also reject
normality → Median (IQR) = 51.40 (42.40–62.17). Note: `age` contains an
implausible max value of 150.4, worth flagging as a likely data-entry error
before this variable is used in any modeling (see A5/A6).

---

### A2. Count Data Fundamentals (10 pts)

**(a) Why OLS fails for `supplement_purchases` (4 pts):**
1. **Range mismatch:** OLS assumes an unbounded, continuous response with
   normally distributed errors. Count data is discrete and bounded at 0, so
   OLS can (and will) predict negative or non-integer purchase counts,
   which are meaningless for this outcome.
2. **Heteroscedasticity:** Count data variance typically scales with the
   mean (as in a Poisson process) rather than staying constant, violating
   OLS's homoscedasticity assumption — this makes OLS standard errors and
   p-values invalid even if the point estimates are unbiased.

**(b) Suitable distribution (6 pts):**
The **Poisson distribution** is the natural baseline: it assumes (i) events
occur independently, (ii) a constant average rate over the exposure period,
and (iii) **equidispersion** — variance equals the mean. Empirically,
`supplement_purchases` has mean = 0.534 and variance = 0.644 (ratio ≈ 1.21),
indicating mild **overdispersion**. When variance exceeds the mean, fitting
a plain Poisson model understates the true standard errors, inflating
z-statistics and producing spuriously significant predictors (Type I error
inflation). The standard remedy is a **Negative Binomial** regression, which
adds a dispersion parameter to absorb the excess variance.

---

### A3. A/B Experimentation (15 pts)

**(a) Hypotheses (3 pts)**
H0: μ_Control = μ_Variant B (no difference in mean daily steps).
H1: μ_Control ≠ μ_Variant B.

**(b) Test execution (8 pts)**
Levene's test confirms equal variances (stat = 0.002, p = 0.965), so the
pooled-variance t-test applies.

| Group | n | Mean | SD |
|---|---|---|---|
| Control | 506 | 3230.00 | 1263.84 |
| Variant B | 494 | 3304.93 | 1287.56 |

**t = −0.9286, df = 998, p = 0.3533.** Cohen's d = 0.059 (negligible effect).

**Decision:** p > 0.05 → **fail to reject H0**. No statistically significant
difference in daily steps between groups.

**Deployment recommendation:** Do not ship Variant B on the strength of this
result — the observed lift is small, statistically indistinguishable from
noise, and the effect size is trivial. With N ≈ 1000 already, a meaningful
real-world effect likely would have been detected; recommend either holding
the current experience or investigating a different lever.

**(c) Type I / Type II in context (4 pts)**
- **Type I (α):** Concluding Variant B improves steps when it doesn't →
  shipping a change with engineering/design cost and no real user benefit.
- **Type II (β):** Concluding no difference when Variant B truly helps →
  forgoing a real engagement improvement, ceding an opportunity to a
  competitor. Given the negligible effect size actually observed, β risk
  here is low — this doesn't look like a "hidden" effect being missed.

---

### A4. Categorical Association (15 pts)

H0: `subscription_tier` and `churned_within_6mo` are independent.
H1: `subscription_tier` and `churned_within_6mo` are associated.

**Observed:**

| Tier | Not churned (0) | Churned (1) |
|---|---|---|
| Basic | 23 | 281 |
| Free | 58 | 453 |
| Premium | 35 | 150 |

**Expected (under independence):**

| Tier | Not churned (0) | Churned (1) |
|---|---|---|
| Basic | 35.26 | 268.74 |
| Free | 59.28 | 451.72 |
| Premium | 21.46 | 163.54 |

**χ² = 14.5198, df = 2, p = 0.000703.**

**Decision:** p < 0.05 → **reject H0**. Tier and churn are significantly
associated.

**Retention insight:** Churn rate is highest for **Basic** (92.4%), then
**Free** (88.6%), lowest for **Premium** (81.1%). Standardized residuals show
Basic is retaining *fewer* users than expected (residual −2.07 in the
not-churned cell) while Premium retains *more* than expected (residual +2.92).
Actionable takeaway: Premium subscribers churn least — retention efforts and
upsell-to-Premium campaigns are the highest-leverage lever, while Basic tier
needs a targeted retention intervention.

---

### A5. Multiple Linear Regression (20 pts)

Model: `daily_steps_avg ~ age + sleep_quality_score`

| Term | Coef | SE | t | p |
|---|---|---|---|---|
| Intercept | 3170.84 | 202.22 | 15.68 | <0.001 |
| age | −22.85 | 2.38 | −9.62 | <0.001 |
| sleep_quality_score | 218.42 | 24.43 | 8.94 | <0.001 |

R² = 0.152, Adj. R² = 0.150, F(2,997) = 89.15, p < 0.001.

**(a) Interpretation (6 pts):**
- **β0 (3170.84):** Predicted daily steps when age = 0 and sleep quality = 0
  — not practically meaningful (extrapolation outside observed data), it
  only anchors the model.
- **β1 (−22.85):** Holding sleep quality constant, each additional year of
  age is associated with **22.85 fewer** average daily steps.
- **β2 (218.42):** Holding age constant, each 1-point increase in sleep
  quality score is associated with **218.42 more** average daily steps —
  the dominant driver of activity in this model.

**(b) Predictor significance (4 pts):** Both partial t-tests are significant
at α = 0.05 (age: t = −9.62, p < 0.001; sleep quality: t = 8.94, p < 0.001)
— both variables contribute independently to explaining steps.

**(c) Multicollinearity & VIF (10 pts):**
Multicollinearity is when two or more predictors are strongly linearly
correlated, making it hard to isolate each predictor's individual effect.
VIF quantifies this by regressing each predictor on all others and computing
1/(1 − R²ⱼ); VIF > 5 (or 10, depending on convention) signals a problem.
Here, VIF(age) = 1.001 and VIF(sleep_quality_score) = 1.001 — essentially no
collinearity. High VIF (not the case here) inflates coefficient standard
errors, widening confidence intervals and making individual t-tests
unreliable even when the overall model fits well; since VIF is ~1 for both
predictors, the standard errors above can be trusted.

---

### A6. Logistic Regression & Classification (25 pts)

Model: `logit(churn) = β0 + β1(age) + β2(is_Premium)`

| Term | Coef | SE | z | p | OR | 95% CI |
|---|---|---|---|---|---|---|
| Intercept | 1.158 | 0.380 | 3.05 | 0.002 | 3.184 | [1.513, 6.700] |
| age | 0.0200 | 0.007 | 2.80 | 0.005 | 1.0202 | [1.006, 1.035] |
| is_Premium | −0.7235 | 0.223 | −3.25 | 0.001 | 0.4850 | [0.314, 0.750] |

**(a) Why logit over linear models (5 pts):** A linear probability model can
predict outside [0, 1], which is nonsensical for a probability, and violates
the constant-variance/normal-error assumptions OLS needs. The logit link
maps a linear combination of predictors onto (0, 1) via the log-odds
transform, which is the natural link for a Bernoulli/binary outcome and
guarantees valid probability predictions.

**(b) Odds ratios (10 pts):**
- **Age (OR = 1.0202):** Each additional year of age is associated with a
  **2.02% increase** in the odds of churning, holding tier constant.
- **Premium tier (OR = 0.4850):** Being on the Premium tier is associated
  with **51.5% lower odds** of churning relative to non-Premium, holding age
  constant. **Yes — Premium tier reduces churn odds** (OR < 1, and the 95%
  CI [0.314, 0.750] excludes 1, confirming significance).

**(c) Model performance (10 pts):**
At the default 0.5 threshold: **Accuracy = 0.884, Precision = 0.884,
Recall = 1.000, ROC-AUC = 0.614**.

Interpretation caveat (important given the class imbalance): the base churn
rate is 88.4%, and at threshold 0.5 the model predicts **every** observation
as "churned" — so Accuracy/Precision/Recall trivially match the base rate
and a perfect recall, not genuine discrimination. **ROC-AUC = 0.614** (only
modestly above the 0.5 chance line) is the more honest measure here and
indicates the model has weak-to-modest discriminatory power; a lower
classification threshold or a different metric (e.g., balanced accuracy,
PR-AUC) would be needed to meaningfully separate churners from retainers in
practice.

---

## Part B: Seaborn `tips` Dataset Analysis (N = 244)

### B1. Distributional Assessment (15 pts)

**(a)** H0: `total_bill` is normally distributed. H1: it is not.

**(b)** **W = 0.9197, p = 3.32 × 10⁻¹⁰** → **reject H0**. Distribution is
right-skewed (skew = 1.133).
- Mean (SD) = 19.79 (8.90)
- **Median (IQR) = 17.80 (13.35 – 24.13)** ← recommended metric, given the
  rejected normality and visible right skew (a few large bills pull the mean
  upward relative to the typical bill).

---

### B2. Party Size Modeling (10 pts)

**(a)** Same logic as A2: `size` is a small discrete count (1–6) with a
natural floor at 1 and no meaningful fractional values. OLS can predict
non-integer or out-of-range party sizes and assumes constant error variance,
which discrete count data with a mean-linked variance typically violates.

**(b)** Poisson assumes **equidispersion** (Var = Mean). Here,
mean(`size`) = 2.570, variance = 0.905 (ratio ≈ 0.35) — this dataset is
actually **underdispersed** relative to Poisson, not overdispersed. The
textbook answer for **overdispersion** (variance > mean) is to switch to
**Negative Binomial** regression; for the underdispersion observed here, the
analogous fix is a model that allows dispersion < 1 (e.g., quasi-Poisson
with an estimated dispersion parameter, or a Conway-Maxwell-Poisson model).

---

### B3. Gender Spend Comparison (15 pts)

**(a)** H0: μ_Male = μ_Female. H1: μ_Male ≠ μ_Female.
Levene's test: p = 0.203 → equal variances assumed.

| Group | n | Mean | SD |
|---|---|---|---|
| Male | 157 | 20.74 | 9.25 |
| Female | 87 | 18.06 | 8.01 |

**t = 2.2778, df = 242, p = 0.0236.**

**Decision:** p < 0.05 → **reject H0**. Male bill payers spend significantly
more on average than female bill payers.

**(b)** **Type I:** Concluding a real gender spend gap exists when it
doesn't → misallocating marketing/staffing based on a false pattern.
**Type II:** Missing a genuine spend difference → forgoing a legitimate
segmentation/upsell opportunity tailored to actual spending behavior.

---

### B4. Analysis of Variance & Independence (15 pts)

**(a) One-way ANOVA: `total_bill ~ day`**

H0: mean total bill is equal across Thur/Fri/Sat/Sun. H1: at least one day
differs.

| Source | SS | df | F | p |
|---|---|---|---|---|
| day | 643.94 | 3 | 2.7675 | 0.0425 |
| Residual | 18614.52 | 240 | | |

**Decision:** p = 0.0425 < 0.05 → **reject H0** (borderline). Mean total bill
differs significantly across days.

**Group means:** Thur 17.68, Fri 17.15, Sat 20.44, Sun 21.41 — bills trend
higher on weekends.

**Recommended post-hoc:** **Tukey HSD**. Running it here, no individual pair
survives the family-wise correction (closest: Sun vs Thur, p = 0.067),
meaning the omnibus signal is real but not strong enough to pin to one
specific day pair — consistent with a gradual weekday→weekend trend rather
than one outlier day.

**(b) Chi-square: `sex` × `smoker`**

H0: sex and smoker status are independent. H1: they are associated.

| | Smoker: Yes | Smoker: No |
|---|---|---|
| Male | 60 | 97 |
| Female | 33 | 54 |

Expected frequencies are nearly identical to observed (Male: 59.84/97.16,
Female: 33.16/53.84). **χ² = 0.000, df = 1, p = 1.00** (with the standard
Yates continuity correction applied to this 2×2 table; uncorrected
χ² = 0.0019, p = 0.965 — same conclusion either way).

**Decision:** p ≫ 0.05 → **fail to reject H0**. No association between sex
and smoking status in this sample.

---

### B5. Tip Prediction Regression (20 pts)

Model: `tip ~ total_bill + size`

| Term | Coef | SE | t | p |
|---|---|---|---|---|
| Intercept | 0.6689 | 0.194 | 3.46 | 0.001 |
| total_bill | 0.0927 | 0.009 | 10.17 | <0.001 |
| size | 0.1926 | 0.085 | 2.26 | 0.025 |

R² = 0.468, Adj. R² = 0.463.

**(a) Interpretation (8 pts):**
- **β1 (0.0927):** Holding party size constant, each **+$1** in total bill is
  associated with a **$0.093** increase in tip.
- **β2 (0.1926):** Holding total bill constant, each **additional guest** is
  associated with a **$0.193** increase in tip.

**(b) Partial t-tests (5 pts):** `total_bill` is highly significant
(t = 10.17, p < 0.001); `size` is significant at α = 0.05 but more marginal
(t = 2.26, p = 0.025).

**(c) VIF & collinearity (7 pts):** VIF(total_bill) = VIF(size) = 1.558.
`total_bill` and `size` are moderately correlated (r = 0.598 — bigger
parties tend to run up bigger bills), but VIF is well below the 5/10
concern threshold, so standard errors and the individual coefficient
estimates above remain trustworthy; collinearity risk is present but not
severe.

---

### B6. High-Tip Binary Logistic Model (25 pts)

Outcome: `high_tip = 1` if `tip ≥ $3.00`, else 0 (base rate 49.6%).
Model: `logit(high_tip) = β0 + β1(total_bill) + β2(smoker)`

**(a) Justification (5 pts):** Same reasoning as A6(a) — a linear
probability model for `high_tip` could predict probabilities outside
[0, 1]; the logit link keeps predictions valid probabilities and correctly
models the binary/Bernoulli outcome.

**(b) Model estimates (10 pts):**

| Term | Coef (β) | SE | z | p |
|---|---|---|---|---|
| Intercept | −3.6747 | 0.534 | −6.89 | <0.001 |
| total_bill | 0.1912 | 0.028 | 6.90 | <0.001 |
| smoker (Yes) | 0.1424 | 0.320 | 0.45 | 0.656 |

**(c) Odds ratios & staff insight (10 pts):**
- **total_bill: OR = 1.2107** [95% CI 1.147, 1.278] — each **+$1** in total
  bill is associated with a **21.1% increase** in the odds of a high tip
  (≥$3), holding smoker status constant. This is the dominant, statistically
  robust driver.
- **smoker: OR = 1.153** [95% CI 0.616, 2.158] — smoker tables show 15.3%
  higher odds of a high tip than non-smokers, but the effect is **not
  statistically significant** (p = 0.656, CI spans 1).

**Actionable insight for staff:** Bill size is the reliable signal for
anticipating a high tip — larger checks warrant continued strong service
focus. Smoking status is **not** a dependable signal and should not factor
into service prioritization.

---

## Reproducibility

All numbers above are produced by [`src/run_analysis.py`](../src/run_analysis.py):

```bash
python3 src/run_analysis.py
```
