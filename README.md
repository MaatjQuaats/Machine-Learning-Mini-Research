# Heart Disease Classification: Paper Reproduction and Extension

This project reproduces the six classifiers and stacking ensemble in Bhagat, Sharma, and Agarwal's heart disease study, then tests how the results change when repeated feature profiles are removed. The extension also compares feature selection, probability calibration, and a decision threshold based on an assumed cost of missed positive cases.

## Files

Place these files in the same folder:

```text
heart.csv
paper_reproduction_and_extension.ipynb
README.md
```

The notebook reads the dataset with `pd.read_csv('heart.csv')`. Run Jupyter from this folder; otherwise change that path in the notebook. The dataset must have the 13 feature columns listed in the notebook and a binary `target` column.

## Requirements

- Python 3.10 or newer
- JupyterLab or Jupyter Notebook
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, and `ipython`

Install the packages with:

```bash
python -m pip install jupyterlab numpy pandas matplotlib seaborn scikit-learn xgboost ipython
```

The saved results were generated with scikit-learn 1.8.0 and XGBoost 3.4.1. Other package versions can change model defaults and results. To match these two versions, install `scikit-learn==1.8.0` and `xgboost==3.4.1` in a compatible environment.

## Run the study

1. Put `heart.csv` beside the notebook.
2. Open the folder in a terminal and run `jupyter lab`.
3. Open `paper_reproduction_and_extension.ipynb`.
4. Select a Python kernel with the packages above installed.
5. Choose **Kernel → Restart Kernel and Run All Cells**. Run the cells from top to bottom; later cells use variables produced earlier.

The notebook displays tables and plots directly. The plots are not saved automatically as image files. For a report, export the relevant notebook figures separately.

## What the notebook does

### Part 1: Reproduce the paper's classifiers

The dataset contains 1,025 rows, 13 input features, and no missing cells. The notebook uses a stratified 80/20 split with seed 42, including 820 rows for training and 205 for testing. `StandardScaler` learns its means and standard deviations from training data only for the five continuous features. The eight coded features are kept as supplied. No imputation is performed because this file has no missing values.

It trains logistic regression (LR), Gaussian naive Bayes (NB), k-nearest neighbours (KNN), decision tree (DT), random forest (RF), and XGBoost (XGB). Five-fold out-of-fold predictions from these six models train a logistic-regression stacking model. It reports accuracy, precision, recall, F1 score, AUC, and confusion matrices, and compares the results with Table 11 of the paper. The code uses a probability threshold of 0.5 for class predictions and probabilities for AUC.

The paper does not provide every setting needed to recover exactly the same fitted models or split. The split seed, scaler, unspecified classifier defaults, and logistic-regression meta-model are documented implementation choices.

### Part 2: Evaluate distinct profiles and decision choices

The file has 723 exact duplicate rows. In Part 1, 202 of 205 test rows have a matching feature profile in the training set. For Part 2, the notebook keeps one row per distinct set of 13 feature values, leaving 302 profiles. Identical feature values do not prove that records belong to the same patient: the file has no patient IDs.

Part 2 uses stratified five-fold cross-validation repeated twice, producing ten outer test evaluations. Within each round, it reserves 25% of the outer training portion for calibration and uses the rest to fit the models. It compares four versions on the **same** outer test fold:

| Version | Features | Probabilities | Decision threshold |
| --- | --- | --- | ---: |
| Baseline stack | All 13 | Original | 0.5 |
| Feature-selected stack | Eight chosen using mutual information on training data | Original | 0.5 |
| Selected + calibrated | Eight | Adjusted using the separate calibration set | 0.5 |
| Proposed method | Eight | Same adjusted probabilities | 1/6 |

Only two stacks are fitted in each round. The final two rows use the same calibrated probabilities; they differ only in their decision thresholds. Feature selection and scaling are fitted within the corresponding training folds. The outer test labels are used only to calculate results.

The proposed threshold assumes that a missed positive (false negative) costs **five times** as much as a false alarm (false positive): `1 / (1 + 5) = 1/6`. This is an experimental assumption, not a cost validated for clinical use. The notebook also checks cost ratios from 1 to 10. It reports the five classification metrics plus Brier score, specificity, false-negative and false-positive counts, assumed cost per profile, and net benefit.

## Main saved results

| Evaluation | Accuracy | Precision | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published stack, Table 11 | 0.9853 | 1.0000 | 0.9727 | 0.9861 | 0.9880 |
| Part 1 reproduced stack, one row-level test split | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Part 2, 13-feature baseline, mean of ten tests | 0.8345 | 0.8300 | 0.8842 | 0.8522 | 0.9057 |
| Part 2, proposed version, mean of ten tests | 0.7765 | 0.7231 | 0.9605 | 0.8236 | 0.8925 |

Under the assumed 5:1 costs, mean cost per distinct test profile decreases from 0.4169 for the Part 2 baseline to 0.3093 for the proposed version. Mean missed positives fall from 3.8 to 1.3 per test fold, while false alarms rise from 6.2 to 12.2. Feature selection and calibration did not improve the baseline independently. The main gain in recall and assumed cost comes from the lower threshold. Accuracy, precision, F1, AUC, and specificity are lower than for the 13-feature Part 2 baseline.

The paper's figure, the Part 1 single-split result, and the Part 2 cross-validation means come from different evaluation setups. They should not be treated as a controlled head-to-head test. The ten repeated folds reuse profiles, so their scores are not ten independent measurements.

## Reference

M. Bhagat, A. Sharma, and P. Agarwal, “An efficient stacking-based ensemble technique for early heart attack prediction,” *Multimedia Tools and Applications*, vol. 84, pp. 36351–36375, 2025, doi: 10.1007/s11042-024-19293-7.
