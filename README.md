# Bank Marketing – Term-Deposit Classification with Five ML Models and a Streamlit App

**Live Streamlit app:** https://ml-classification-app-jdlovav7sl9wtahkftmydt.streamlit.app/
**GitHub repository:** https://github.com/athisayarajp/ml-classification-streamlit.git 

## 1. Problem statement
A bank runs telephone marketing campaigns and wants to know which clients will subscribe to a term deposit. This project builds and compares **five classification models** (Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes and Support Vector Machine) that predict the binary target `y` (`yes` = subscribed, `no` = did not subscribe) from client, campaign and economic attributes. The models are evaluated with six metrics (Accuracy, AUC, Precision, Recall, F1, MCC), saved as pipelines, and served in an interactive Streamlit app that evaluates an uploaded test CSV **without retraining**.

## 2. Dataset description
| Item | Value |
|---|---|
| Name | Bank Marketing – `bank-additional-full.csv` |
| Source | UCI Machine Learning Repository – https://archive.ics.uci.edu/dataset/222/bank+marketing |
| Citation | S. Moro, P. Cortez, P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 2014. |
| Instances | 41,188 (41,176 after removing 12 duplicate rows) – minimum required: 500 |
| Features | 20 (10 numerical, 10 categorical) – minimum required: 12 |
| Target | `y` – client subscribed to a term deposit (`yes` ≈ 11.3%, `no` ≈ 88.7%) |

- **Numerical features:** `age`, `duration`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`.
- **Categorical features:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`.
- **Data notes:** no null values, but several categorical columns use `'unknown'` as a placeholder (kept as its own category). `pdays = 999` means "not previously contacted". `duration` (call length) is only known after a call ends, so it would not be available for a real pre-call prediction; it is kept because it is part of the assignment dataset.

### Preprocessing methodology
1. Removed 12 exact duplicate rows before splitting, so no record can appear in both train and test.
2. Stratified 80/20 train/test split (`random_state=42`): 32,940 training rows and 8,236 test rows, keeping the ~11% `yes` rate in both.
3. A `ColumnTransformer` inside every model `Pipeline`: `StandardScaler` for numerical columns and `OneHotEncoder(handle_unknown='ignore')` for categorical columns. It is fitted on training data only, so there is no data leakage. Gaussian Naive Bayes uses the same steps with a dense encoder output.
4. Each saved model is a complete pipeline (preprocessing + classifier), so the app accepts raw feature columns.

## 3. GitHub repository link
https://github.com/athisayarajp/ml-classification-streamlit.git

## 4. Models used
Logistic Regression · Decision Tree · K-Nearest Neighbors (k=5) · Gaussian Naive Bayes · Support Vector Machine (RBF kernel). All five were trained on the same training split with default hyper-parameters (`random_state=42` where applicable) and evaluated on the same 8,236-row test set. The positive class is `yes`. AUC for the SVM is computed from its decision-function scores.

### Comparison table (test set)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.9095 | **0.9390** | 0.6522 | 0.4224 | 0.5128 | 0.4788 |
| Decision Tree | 0.8902 | 0.7246 | 0.5130 | 0.5108 | 0.5119 | 0.4500 |
| K-Nearest Neighbors | 0.9014 | 0.8741 | 0.5858 | 0.4267 | 0.4938 | 0.4474 |
| Naive Bayes | 0.8227 | 0.8404 | 0.3506 | **0.6724** | 0.4609 | 0.3950 |
| Support Vector Machine | **0.9119** | 0.9163 | **0.6683** | 0.4321 | **0.5249** | **0.4926** |

Bold = best value for that metric.

### Observations on model performance
| ML Model Name | Observation |
|:--|:--|
| Logistic Regression | Strong, simple baseline: best AUC (0.939) and second-best accuracy, F1 and MCC. Recall for `yes` is low (0.42) at the default 0.5 threshold. |
| Decision Tree | Overfits (100% train vs 89% test accuracy) and has the weakest AUC (0.725). Recall (0.51) is second-best, but precision is low. |
| K-Nearest Neighbors | Middling on every metric (F1 0.49, MCC 0.45); one-hot features make distances less informative and k=5 favours the majority class. |
| Naive Bayes | Highest recall (0.67) but lowest precision (0.35), accuracy (0.82) and MCC (0.40): the independence assumption does not hold for these correlated features. |
| Support Vector Machine | Best accuracy (0.912), precision (0.668), F1 (0.525) and MCC (0.493); AUC (0.916) is second to Logistic Regression. |

**Conclusion.** No single model wins every metric. The SVM gives the best overall balance (highest MCC and F1); Logistic Regression is a close second and the better choice if interpretability, speed or ranking quality matter more; Naive Bayes is preferable only if finding as many subscribers as possible matters more than wasted calls. Because the target is imbalanced, accuracy alone is misleading (always predicting `no` scores 88.7%), so MCC and F1 are the more informative metrics. The SVM-vs-Logistic Regression differences are small and come from a single split, so they are not a proven difference.

## 5. Streamlit app and no-retraining deployment
The app (`app.py`) loads the saved pipelines from `model/*.joblib` and **never calls `fit()`**. Workflow: upload CSV → validate required columns → select a model → predict → compute the six metrics → show the confusion matrix (counts and row %) and classification report. An optional button compares all five models on the uploaded data. `test_data.csv` (the 8,236 held-out rows with target `y`) can be downloaded from the app and uploaded back to reproduce the table above.

The saved models only load reliably with the scikit-learn version they were created with, so `requirements.txt` pins the exact versions recorded in `model/versions.txt` (Python 3.13, scikit-learn 1.6.1, pandas 2.2.3, numpy 2.4.6).

## 6. Repository structure
```
ml-classification-streamlit/
├── app.py                    # Streamlit application
├── requirements.txt          # pinned dependencies
├── README.md
├── test_data.csv             # held-out test data (with target y)
├── ML_Assignment2.ipynb      # executed notebook: EDA, preprocessing, 5 models, comparison, model export
├── CHECKLIST.md              # milestone checklist
├── data/
│   └── bank-additional-full.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── versions.txt
```

## 7. Run locally
```bash
python -m venv .venv            # Python 3.13 recommended
# activate the environment, then:
pip install -r requirements.txt
streamlit run app.py
```
Then upload `test_data.csv` in the sidebar. To regenerate the models and `test_data.csv`, run all cells of `ML_Assignment2.ipynb` from the repository root.
