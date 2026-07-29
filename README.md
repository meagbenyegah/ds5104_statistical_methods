# DS5104 — Statistical Methods for Data Science: Final Assessment

Statistical analysis for the DS5104 final assessment (Academic City University,
Accra). Part A analyzes `data/fitpulse_health_dataset.csv`; Part B analyzes
`seaborn`'s built-in `tips` dataset.

## Setup

Requires Python 3.10+.

```bash
# from the project root
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the analysis script

Prints every test/model result (Shapiro-Wilk, t-tests, chi-square, ANOVA +
Tukey HSD, OLS + VIF, logistic regression + odds ratios/ROC-AUC) for both
parts to the terminal.

```bash
python3 src/run_analysis.py
```

## Run the notebook

`notebooks/DS5104_Final_Assessment_Solution.ipynb` contains the same
computations as the script, interleaved with the write-up and a chart per
question.

**Interactively** (opens in your browser):

```bash
jupyter lab notebooks/DS5104_Final_Assessment_Solution.ipynb
# or: jupyter notebook notebooks/DS5104_Final_Assessment_Solution.ipynb
```
Then Kernel → Restart & Run All.

**Or non-interactively**, to re-execute it in place from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/DS5104_Final_Assessment_Solution.ipynb
```

Both run against a relative `../data/fitpulse_health_dataset.csv` path, so
run them from `notebooks/`'s working directory as Jupyter sets it (i.e. don't
move the notebook file without adjusting the path).

## Project structure

- `docs/FINAL_EXAMS_STAT_METHODS_DATA_SCIENCE_26_IGIT.{md,pdf}` — the exam brief.
- `docs/EXAM_ANSWER_PLAN.md` — methodology plan written before the dataset was available.
- `docs/EXAM_ANSWERS.md` — full write-up: statistics, hypotheses, decisions, interpretations.
- `data/fitpulse_health_dataset.csv` — Part A dataset (1000 rows).
- `src/run_analysis.py` — reproducible analysis script.
- `notebooks/DS5104_Final_Assessment_Solution.ipynb` — runnable notebook version with charts.
- `requirements.txt` — pinned dependency versions.

See `CLAUDE.md` for current progress status and a known data-quality issue in
the `age` column.
