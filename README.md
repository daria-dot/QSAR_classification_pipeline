# QSAR Biodegradation Classifier

Predicts whether a molecule is readily biodegradable (RB) or not (NRB) from its
molecular descriptors, using a stacked ensemble over the UCI QSAR biodegradation
dataset. It runs end to end and produces every figure and table in the report.

## How it works

Three base learners look at the data in different ways, and a logistic
regression on top decides how much to trust each one:

| Branch | Feature selection | Model |
| --- | --- | --- |
| 1 | Boruta (random-forest relevance) | HistGradientBoosting |
| 2 | LASSO (L1 logistic), on scaled features | Linear SVM |
| 3 | none, scaled features only | Distance-weighted KNN |

Boruta is fit once on the outer training fold and then frozen inside the stack,
so the selection never sees validation data. The stacking classifier uses 5-fold
stratified CV internally to build the meta-features.

Both selectors have fallbacks for the awkward cases that show up in small CV
folds: a single-class fold, or a selection that comes back nearly empty. When
that happens they either keep the top-k features by importance or pass
everything through, and they say so in the log.

Scoring happens twice. First a 3-fold outer cross-validation for an honest F1
estimate, then a single 80/20 split to fit the model the figures are drawn from.

## You need the dataset

`QSAR_data.mat` is not in this repo. Drop it in the project root before running.
The loader picks the largest variable in the `.mat` file, treats the last column
as the 0/1 label (0 = NRB, 1 = RB) and everything before it as descriptors.

## Setup and run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy boruta
python main_run.py
```

Written against Python 3.11+. Boruta plus a nested stack is the slow part, so
expect this to sit there for a few minutes rather than finish instantly.

## What you get

Everything lands in the working directory:

- `fig1_confusion_matrix.pdf`
- `fig2_roc_curve.pdf`
- `fig3_pr_curve.pdf`
- `fig4_learning_curve.pdf` (F1 against training size)
- `fig5_meta_weights.pdf` (how the meta-learner weighs each branch)
- `fig6_branch_roc_comparison.pdf`
- `table_comparison.tex` (ensemble vs. each base learner, ready to paste into LaTeX)

Test-set AUC, F1, sensitivity, specificity and a per-branch breakdown are
printed to stdout as it goes.

## Results

Full write-up is in [`report.pdf`](report.pdf): *Hybrid Linear and Non-Linear
Feature Selection for QSAR Classification using a Stacked Ensemble*. It covers
the dataset (1,055 compounds, 41 descriptors), the architecture, and the
analysis behind each figure.

Headline numbers, on the held-out 20% test set:

| Model | AUC | F1 | Sensitivity | Specificity |
| --- | --- | --- | --- | --- |
| **Stacked ensemble** | 0.941 | **0.844** | 0.873 | 0.900 |
| HGB (Boruta) | **0.947** | 0.831 | 0.831 | 0.914 |
| SVM (LASSO) | 0.915 | 0.775 | 0.873 | 0.807 |
| KNN (full features) | 0.917 | 0.816 | 0.845 | 0.886 |

HGB alone edges the ensemble on AUC, but it misses more biodegradable
compounds. The stack trades a little of that for the best F1 and the most even
sensitivity/specificity split. The meta-learner leans on HGB and KNN
(coefficients 1.65 and 1.62) and weighs SVM lower at 0.42, though the SVM branch
still earns its place by making different mistakes than the other two.

These are the figures from the run reported in the paper. Seeds are fixed at 42
throughout, so a rerun should land in the same place, give or take whatever your
scikit-learn version does differently.

## Layout

| File | What's in it |
| --- | --- |
| `main_run.py` | The script that runs everything. Start here. |
| `data_loader.py` | Reads the `.mat` file into a DataFrame |
| `pre_process.py` | Sanity checks, drops duplicate rows and duplicate descriptors |
| `boruta_class.py` | Boruta wrapped as an sklearn transformer |
| `ensemble_model.py` | Builds the stack, plus the LASSO selector |
| `evaluate_base_learners.py` | Scores each branch on its own |
| `plots.py` | All the figures and the LaTeX table |

## If it breaks

`AttributeError: module 'numpy' has no attribute 'float'` from inside BorutaPy
means your numpy is too new for the released `boruta` package. Either pin
`numpy<1.24` or install Boruta from its git master.
