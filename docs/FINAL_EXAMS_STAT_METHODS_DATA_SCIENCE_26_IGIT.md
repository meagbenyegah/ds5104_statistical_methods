ACADEMIC CITY UNIVERSITY
FACULTY OF COMPUTATIONAL AND INFORMATICS
ACCRA-GHANA

DS5104: STATISTICAL METHODS FOR DATA SCIENCE

DEADLINE: 29/07/2026

FINAL ASSESSMENT

INDEX NUMBER:

SECTION A: GENERAL INSTRUCTIONS & GUIDELINES

• Answer all questions in both Part A and Part B.

• Use appropriate statistical software (e.g., Python or R) where computation is required.

• Clearly state all null (H0) and alternative (H1) hypotheses prior to conducting statistical tests.

• All hypothesis tests must be conducted at a significance level of α = 0.05 unless explicitly stated otherwise.

• Provide formal statistical interpretations alongside numerical outputs (test statistics, degrees of freedom, and

p-values).

Part A: FitPulse Health Dataset Analysis

1. (15 points) Normality & Descriptive Metrics

Using the variables daily steps avg and age from fitpulse health dataset.csv, evaluate distribu-
tional characteristics.

(a) (4 points) Formulate H0 and H1 for testing the normality of daily steps avg via the Shapiro-Wilk test.

(b) (11 points) Perform the Shapiro-Wilk test at α = 0.05:

1. Report the test statistic (W ) and corresponding p-value.

2. State your statistical decision regarding H0.

3. Determine whether to summarize the variable using Mean (SD) or Median (IQR).

4. Justify your metric choice based on distribution characteristics.

2. (10 points) Count Data Fundamentals

The outcome supplement purchases measures non-negative annual user orders.

(a) (4 points) Explain two primary theoretical reasons why OLS linear regression fails for non-negative count

data.

(b) (6 points) Identify a suitable count distribution, state its core assumptions, and explain how overdispersion

invalidates it.

3. (15 points) A/B Experimentation

An A/B test (ab test group: Control vs. Variant B) measures impact on daily steps avg.

(a) (3 points) State H0 and H1 for an Independent Two-Sample t-test comparing mean step counts between

groups.

(b) (8 points) Execute the test, reporting test statistic (t), df , p-value, statistical decision, and a deployment rec-

ommendation.

(c) (4 points) Contextualize Type I (α) and Type II (β) errors in terms of business impact for this product launch.

4. (15 points) Categorical Association

Evaluate customer churn across subscription tiers (subscription tier: Free, Basic, Premium).

(a) (15 points) Conduct a Chi-Square (χ2) Test of Independence against churned within 6mo. State hy-
potheses, present the expected frequencies table, and report df , χ2 statistic, p-value, and retention insights.

5. (20 points) Multiple Linear Regression

Consider the multiple regression model predicting daily activity:

(cid:100)steps = ˆβ0 + ˆβ1(age) + ˆβ2(sleep quality)

(a) (6 points) Fit the model and interpret coefficients ˆβ0, ˆβ1, and ˆβ2 in context.

(b) (4 points) Assess predictor significance using partial t-tests and p-values.

(c) (10 points) Define multicollinearity, explain how Variance Inflation Factor (VIF) detects it, and discuss its

impact on standard errors.

6. (25 points) Logistic Regression & Classification

Model churn probability (churned within 6mo: 1 = Yes, 0 = No):

ln

(cid:18) p

(cid:19)

1 − p

= β0 + β1(age) + β2(Premium Tier)

(a) (5 points) Explain why the logit link function is preferred over linear models for binary outcomes.

(b) (10 points) Compute and interpret Odds Ratios (OR = eβ) for both predictors. Clarify if Premium tier reduces

churn odds.

(c) (10 points) Evaluate overall model performance using Accuracy, Precision, Recall, and ROC-AUC metrics.

Part B: Seaborn tips Dataset Analysis (N = 244)

1. (15 points) Distributional Assessment

Analyze total dining expenditure (total bill).

(a) (5 points) Formulate hypotheses and perform a Shapiro-Wilk test on total bill.

(b) (10 points) Report statistic (W ), p-value, and decision. Justify choosing Mean (SD) vs. Median (IQR).

2. (10 points) Party Size Modeling

Analyze customer party size (size).

(a) (5 points) Explain why OLS regression is unsuitable for discrete party counts like size.

Page 2

(b) (5 points) Describe the Poisson assumption of equidispersion and state alternative models when variance

exceeds the mean.

3. (15 points) Gender Spend Comparison

Compare spending between male and female bill payers (sex vs. total bill).

(a) (10 points) State hypotheses and conduct an Independent Two-Sample t-test (t, df , p-value, decision).

(b) (5 points) Explain the practical consequences of committing a Type I vs. Type II error for restaurant manage-

ment.

4. (15 points) Analysis of Variance & Independence

(a) (8 points) Conduct a One-Way ANOVA testing mean total bill across operating days (Thur, Fri, Sat,
Sun). Provide hypotheses, ANOVA table summary (F , df , p-value), conclusion, and recommended post-hoc
test.

(b) (7 points) Conduct a Chi-Square (χ2) Test of Independence between sex and smoker (df , χ2, p-value,

conclusion).

5. (20 points) Tip Prediction Regression
Consider the tip prediction model:

(cid:99)tip = β0 + β1(total bill) + β2(size)

(a) (8 points) Fit the model and interpret slope parameters ˆβ1 and ˆβ2.

(b) (5 points) Evaluate individual predictor significance using partial t-tests.

(c) (7 points) Compute VIF values for predictors and discuss collinearity risk between bill size and table count.

6. (25 points) High-Tip Binary Logistic Model

Define binary outcome high tip = 1 if tip ≥ $3.00, else 0. Fit logit model:

ln

(cid:18) p

(cid:19)

1 − p

= β0 + β1(total bill) + β2(smoker)

(a) (5 points) Justify logistic regression over linear probability models for high tip.
(b) (10 points) Report model estimates: coefficients ( ˆβ), standard errors (SE), z-scores, and p-values.
(c) (10 points) Derive and interpret Odds Ratios (e ˆβ) for total bill and smoker with actionable staff in-

sights.

Page 3


