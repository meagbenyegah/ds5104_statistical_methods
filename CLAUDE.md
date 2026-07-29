# Louisa — DS5104 Final Assessment

Statistical analysis project for the DS5104: Statistical Methods for Data
Science final assessment (Academic City University, Accra). Deadline:
2026-07-29.

## Structure

- `docs/FINAL_EXAMS_STAT_METHODS_DATA_SCIENCE_26_IGIT.{md,pdf}` — the exam
  brief (Part A: FitPulse Health Dataset, 6 questions; Part B: seaborn `tips`
  dataset, 6 questions).
- `docs/EXAM_ANSWER_PLAN.md` — methodology plan (which test/model per
  question, what to report) written before the dataset was available.
- `docs/EXAM_ANSWERS.md` — full write-up with computed statistics,
  hypotheses, decisions, and interpretations for all 12 questions.
- `data/fitpulse_health_dataset.csv` — Part A dataset (1000 rows; custom/
  synthetic, built for this course — not a public Kaggle dataset).
- `src/run_analysis.py` — reproducible script that runs every test/model for
  both parts (Shapiro-Wilk, t-tests, chi-square, ANOVA + Tukey HSD, OLS +
  VIF, logistic regression + odds ratios/ROC-AUC). Part B pulls
  `seaborn.load_dataset("tips")` directly, no file needed.
- `notebooks/DS5104_Final_Assessment_Solution.ipynb` — runnable notebook
  version: same computations as `src/run_analysis.py`, interleaved with the
  write-up from `docs/EXAM_ANSWERS.md` and a chart per question.
- `requirements.txt` — pinned minimum versions for the analysis + notebook
  tooling (pandas, numpy, scipy, statsmodels, scikit-learn, seaborn,
  matplotlib, jupyter, nbformat, nbconvert, ipykernel).

## Progress

- [x] Exam brief transcribed to Markdown (`docs/FINAL_EXAMS_STAT_METHODS_DATA_SCIENCE_26_IGIT.md`)
- [x] Methodology plan drafted (`docs/EXAM_ANSWER_PLAN.md`)
- [x] Dataset confirmed not public (checked Kaggle — no match)
- [x] Dataset placed at `data/fitpulse_health_dataset.csv`
- [x] Python env set up (pandas, scipy, statsmodels, scikit-learn, seaborn)
- [x] Full analysis run for Part A (6 questions) and Part B (6 questions)
- [x] Answers compiled into `docs/EXAM_ANSWERS.md`
- [ ] INDEX NUMBER filled in on `docs/EXAM_ANSWERS.md`
- [ ] Final review / submission

## Known data issue

`age` in `fitpulse_health_dataset.csv` has an implausible max value (150.4),
likely a data-entry error. Flagged in A1/A5/A6 discussion but left uncleaned
since the assignment doesn't ask for data cleaning — mention this if asked to
revisit age-based analysis, and offer a sensitivity re-run excluding it.

## Running the analysis

```bash
pip install -r requirements.txt
python3 src/run_analysis.py
# or, for the notebook:
jupyter nbconvert --to notebook --execute --inplace notebooks/DS5104_Final_Assessment_Solution.ipynb
```
