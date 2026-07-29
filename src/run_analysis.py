import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import seaborn as sns

pd.set_option('display.width', 200)
SEP = "=" * 70


def header(title):
    print("\n" + SEP)
    print(title)
    print(SEP)


# ---------------------------------------------------------------
# PART A: FitPulse Health Dataset
# ---------------------------------------------------------------
fp = pd.read_csv("data/fitpulse_health_dataset.csv")
fp["is_premium"] = (fp["subscription_tier"] == "Premium").astype(int)

header("A1. Shapiro-Wilk: daily_steps_avg")
w, p = stats.shapiro(fp["daily_steps_avg"])
print(f"W = {w:.4f}, p = {p:.6g}")
print(f"Mean = {fp['daily_steps_avg'].mean():.2f}, SD = {fp['daily_steps_avg'].std():.2f}")
med = fp["daily_steps_avg"].median()
q1, q3 = fp["daily_steps_avg"].quantile([0.25, 0.75])
print(f"Median = {med:.2f}, IQR = {q1:.2f} - {q3:.2f} ({q3-q1:.2f})")
print(f"Skew = {fp['daily_steps_avg'].skew():.3f}, Kurtosis = {fp['daily_steps_avg'].kurt():.3f}")

header("A1. Shapiro-Wilk: age (supporting)")
w2, p2 = stats.shapiro(fp["age"])
print(f"W = {w2:.4f}, p = {p2:.6g}")
print(f"Mean = {fp['age'].mean():.2f}, SD = {fp['age'].std():.2f}")
med_a = fp["age"].median()
q1a, q3a = fp["age"].quantile([0.25, 0.75])
print(f"Median = {med_a:.2f}, IQR = {q1a:.2f} - {q3a:.2f} ({q3a-q1a:.2f})")
print(f"min={fp['age'].min()}, max={fp['age'].max()}  <-- note implausible max, flag as data quality issue")

header("A2. supplement_purchases: mean vs variance (overdispersion check)")
m = fp["supplement_purchases"].mean()
v = fp["supplement_purchases"].var()
print(f"Mean = {m:.4f}, Variance = {v:.4f}, Variance/Mean ratio = {v/m:.3f}")

header("A3. Independent t-test: daily_steps_avg by ab_test_group")
ctrl = fp.loc[fp.ab_test_group == "Control", "daily_steps_avg"]
var_b = fp.loc[fp.ab_test_group == "Variant_B", "daily_steps_avg"]
lev_stat, lev_p = stats.levene(ctrl, var_b)
print(f"Levene's test for equal variances: stat={lev_stat:.4f}, p={lev_p:.4f}")
equal_var = lev_p >= 0.05
t_stat, t_p = stats.ttest_ind(ctrl, var_b, equal_var=equal_var)
df_t = len(ctrl) + len(var_b) - 2 if equal_var else None
print(f"n_control={len(ctrl)}, mean={ctrl.mean():.2f}, sd={ctrl.std():.2f}")
print(f"n_variantB={len(var_b)}, mean={var_b.mean():.2f}, sd={var_b.std():.2f}")
print(f"equal_var assumption used: {equal_var}")
print(f"t = {t_stat:.4f}, df = {df_t}, p = {t_p:.6g}")
pooled_sd = np.sqrt(((len(ctrl)-1)*ctrl.var() + (len(var_b)-1)*var_b.var()) / (len(ctrl)+len(var_b)-2))
cohens_d = (var_b.mean() - ctrl.mean()) / pooled_sd
print(f"Cohen's d = {cohens_d:.4f}")

header("A4. Chi-square: subscription_tier x churned_within_6mo")
tbl = pd.crosstab(fp.subscription_tier, fp.churned_within_6mo)
tbl = tbl[[0, 1]]
print("Observed:\n", tbl)
chi2, p_chi, dof, expected = stats.chi2_contingency(tbl)
exp_df = pd.DataFrame(expected, index=tbl.index, columns=tbl.columns)
print("\nExpected:\n", exp_df.round(2))
print(f"\nchi2 = {chi2:.4f}, df = {dof}, p = {p_chi:.6g}")
churn_rate = fp.groupby("subscription_tier")["churned_within_6mo"].mean().sort_values()
print("\nChurn rate by tier:\n", churn_rate.round(3))
std_resid = (tbl - exp_df) / np.sqrt(exp_df)
print("\nStandardized residuals:\n", std_resid.round(2))

header("A5. Multiple Linear Regression: daily_steps_avg ~ age + sleep_quality_score")
m5 = smf.ols("daily_steps_avg ~ age + sleep_quality_score", data=fp).fit()
print(m5.summary())
X5 = sm.add_constant(fp[["age", "sleep_quality_score"]])
vif5 = pd.DataFrame({
    "variable": X5.columns,
    "VIF": [variance_inflation_factor(X5.values, i) for i in range(X5.shape[1])]
})
print("\nVIF:\n", vif5)

header("A6. Logistic Regression: churned_within_6mo ~ age + is_premium")
X6 = sm.add_constant(fp[["age", "is_premium"]])
y6 = fp["churned_within_6mo"]
m6 = sm.Logit(y6, X6).fit(disp=0)
print(m6.summary())
orA = np.exp(m6.params)
ci = m6.conf_int()
ci_exp = np.exp(ci)
print("\nOdds Ratios:\n", pd.DataFrame({"OR": orA, "CI_low": ci_exp[0], "CI_high": ci_exp[1]}))

pred_prob = m6.predict(X6)
pred_class = (pred_prob >= 0.5).astype(int)
print(f"\nAccuracy  = {accuracy_score(y6, pred_class):.4f}")
print(f"Precision = {precision_score(y6, pred_class, zero_division=0):.4f}")
print(f"Recall    = {recall_score(y6, pred_class):.4f}")
print(f"ROC-AUC   = {roc_auc_score(y6, pred_prob):.4f}")
print(f"Predicted class distribution: {pd.Series(pred_class).value_counts().to_dict()}  (base rate churned={y6.mean():.3f})")


# ---------------------------------------------------------------
# PART B: Seaborn tips dataset
# ---------------------------------------------------------------
tips = sns.load_dataset("tips")
tips["high_tip"] = (tips["tip"] >= 3.00).astype(int)
tips["is_smoker"] = (tips["smoker"] == "Yes").astype(int)

header(f"Part B dataset check: N = {len(tips)}")
print(tips.head())

header("B1. Shapiro-Wilk: total_bill")
wb, pb = stats.shapiro(tips["total_bill"])
print(f"W = {wb:.4f}, p = {pb:.6g}")
print(f"Mean = {tips['total_bill'].mean():.2f}, SD = {tips['total_bill'].std():.2f}")
medb = tips["total_bill"].median()
q1b, q3b = tips["total_bill"].quantile([0.25, 0.75])
print(f"Median = {medb:.2f}, IQR = {q1b:.2f} - {q3b:.2f} ({q3b-q1b:.2f})")
print(f"Skew = {tips['total_bill'].skew():.3f}")

header("B2. size: mean vs variance (equidispersion check)")
ms = tips["size"].mean()
vs = tips["size"].var()
print(f"Mean = {ms:.4f}, Variance = {vs:.4f}, Variance/Mean ratio = {vs/ms:.3f}")

header("B3. Independent t-test: total_bill by sex")
male = tips.loc[tips.sex == "Male", "total_bill"]
female = tips.loc[tips.sex == "Female", "total_bill"]
lev_stat_b, lev_p_b = stats.levene(male, female)
print(f"Levene's test: stat={lev_stat_b:.4f}, p={lev_p_b:.4f}")
equal_var_b = lev_p_b >= 0.05
t_stat_b, t_p_b = stats.ttest_ind(male, female, equal_var=equal_var_b)
df_tb = len(male) + len(female) - 2 if equal_var_b else None
print(f"n_male={len(male)}, mean={male.mean():.2f}, sd={male.std():.2f}")
print(f"n_female={len(female)}, mean={female.mean():.2f}, sd={female.std():.2f}")
print(f"equal_var assumption used: {equal_var_b}")
print(f"t = {t_stat_b:.4f}, df = {df_tb}, p = {t_p_b:.6g}")

header("B4a. One-way ANOVA: total_bill ~ day")
groups = [g["total_bill"].values for _, g in tips.groupby("day", observed=True)]
day_names = [n for n, _ in tips.groupby("day", observed=True)]
f_stat, f_p = stats.f_oneway(*groups)
m4 = smf.ols("total_bill ~ C(day)", data=tips).fit()
anova_tbl = sm.stats.anova_lm(m4, typ=2)
print(anova_tbl)
print(f"\nF = {f_stat:.4f}, p = {f_p:.6g}")
print("\nGroup means:\n", tips.groupby("day", observed=True)["total_bill"].agg(["mean", "std", "count"]))

if f_p < 0.05:
    print("\nTukey HSD post-hoc:")
    tukey = pairwise_tukeyhsd(tips["total_bill"], tips["day"])
    print(tukey)
else:
    print("\nANOVA not significant at alpha=0.05 -> Tukey HSD run for completeness:")
    tukey = pairwise_tukeyhsd(tips["total_bill"], tips["day"])
    print(tukey)

header("B4b. Chi-square: sex x smoker")
tbl_b = pd.crosstab(tips.sex, tips.smoker)
print("Observed:\n", tbl_b)
chi2_b, p_chi_b, dof_b, expected_b = stats.chi2_contingency(tbl_b)
print("\nExpected:\n", pd.DataFrame(expected_b, index=tbl_b.index, columns=tbl_b.columns).round(2))
print(f"\nchi2 = {chi2_b:.4f}, df = {dof_b}, p = {p_chi_b:.6g}")

header("B5. Multiple Linear Regression: tip ~ total_bill + size")
m5b = smf.ols("tip ~ total_bill + size", data=tips).fit()
print(m5b.summary())
X5b = sm.add_constant(tips[["total_bill", "size"]])
vif5b = pd.DataFrame({
    "variable": X5b.columns,
    "VIF": [variance_inflation_factor(X5b.values, i) for i in range(X5b.shape[1])]
})
print("\nVIF:\n", vif5b)
print(f"\nCorrelation(total_bill, size) = {tips['total_bill'].corr(tips['size']):.4f}")

header("B6. Logistic Regression: high_tip ~ total_bill + is_smoker")
X6b = sm.add_constant(tips[["total_bill", "is_smoker"]])
y6b = tips["high_tip"]
m6b = sm.Logit(y6b, X6b).fit(disp=0)
print(m6b.summary())
orB = np.exp(m6b.params)
ciB = m6b.conf_int()
ciB_exp = np.exp(ciB)
print("\nOdds Ratios:\n", pd.DataFrame({"OR": orB, "CI_low": ciB_exp[0], "CI_high": ciB_exp[1]}))
print(f"\nhigh_tip base rate: {y6b.mean():.3f}")
