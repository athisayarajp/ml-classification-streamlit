"""Streamlit app - evaluate five saved classification pipelines on an uploaded test CSV.

The models are loaded from model/*.joblib and are NEVER retrained here (no fit() is called).
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SAMPLE_CSV = BASE_DIR / "test_data.csv"
TARGET = "y"
LABELS = ["no", "yes"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
MODEL_INFO = {
    "Logistic Regression": "Linear model; outputs class probabilities. Strong baseline and best AUC in the experiments.",
    "Decision Tree": "Rule-based tree (unrestricted depth). Easy to interpret but overfits the training data.",
    "K-Nearest Neighbors": "Predicts by majority vote of the 5 nearest training clients (scaled features).",
    "Naive Bayes": "Gaussian Naive Bayes; assumes independent features. Highest recall, lowest precision.",
    "Support Vector Machine": "RBF-kernel SVM on scaled features. Best accuracy, precision, F1 and MCC in the experiments.",
}

st.set_page_config(page_title="Bank Marketing - Model Evaluation", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Loading saved model...")
def load_model(file_name: str):
    """Load a saved pipeline from disk (no training)."""
    return joblib.load(MODEL_DIR / file_name)


def get_scores(model, X):
    """Positive-class probability if available, otherwise the SVM decision score."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    auc = roc_auc_score(y_true, y_score) if y_true.nunique() == 2 else float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks([0, 1], labels=[f"Pred {l}" for l in LABELS])
    ax.set_yticks([0, 1], labels=[f"Actual {l}" for l in LABELS])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({row_pct[i, j]:.1f}%)", ha="center", va="center",
                    color="white" if row_pct[i, j] > 50 else "black", fontsize=11)
    ax.set_title("Confusion matrix (count and row %)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- sidebar
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("1. Upload test CSV", type="csv",
                                    help="Must contain the 20 feature columns and the target column 'y'.")
use_sample = st.sidebar.checkbox("...or use the bundled test_data.csv", value=False)
model_name = st.sidebar.selectbox("2. Select model", list(MODEL_FILES))
st.sidebar.caption(MODEL_INFO[model_name])
if SAMPLE_CSV.exists():
    st.sidebar.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")
st.sidebar.divider()
st.sidebar.caption(f"scikit-learn {sklearn.__version__} | pandas {pd.__version__}")

# -------------------------------------------------------------------- main
st.title("📊 Bank Marketing - Classification Model Evaluation")
st.write("Predict whether a bank client subscribes to a term deposit. Upload a test CSV, choose one of five "
         "trained models and view the six evaluation metrics, confusion matrix and classification report.")
st.info("The uploaded data is only used for **evaluation**. The models are loaded from saved files "
        "and are **not retrained**.")

try:
    model = load_model(MODEL_FILES[model_name])
except Exception as exc:  # e.g. scikit-learn version mismatch
    st.error(f"Could not load the saved model '{MODEL_FILES[model_name]}'. Check that the scikit-learn "
             f"version in requirements.txt matches the one used for training.\n\nDetails: {exc}")
    st.stop()

required = list(model.feature_names_in_)
with st.expander("Expected CSV format"):
    st.write(f"The file needs these **{len(required)} feature columns** plus the target column "
             f"`{TARGET}` (values `yes` / `no`):")
    st.code(", ".join(required) + f", {TARGET}")

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample and SAMPLE_CSV.exists():
    df = pd.read_csv(SAMPLE_CSV)
else:
    st.warning("Upload a CSV in the sidebar (or tick the bundled test data) to begin.")
    st.stop()

# ----- validation
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required column(s): {', '.join(missing)}")
    st.stop()
if TARGET not in df.columns:
    st.error(f"The target column '{TARGET}' is missing; it is needed to compute the metrics.")
    st.stop()

y_raw = df[TARGET].astype(str).str.strip().str.lower().map({"no": 0, "yes": 1, "0": 0, "1": 1})
if y_raw.isna().any():
    st.error(f"Column '{TARGET}' must contain only yes/no (or 0/1) values.")
    st.stop()
y_true = y_raw.astype(int)
X = df[required]
extra = [c for c in df.columns if c not in required + [TARGET]]
if extra:
    st.caption(f"Ignored extra columns: {', '.join(extra)}")

with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20), width="stretch")
    st.caption(f"{len(df):,} rows | {len(required)} features | actual 'yes' rate: {y_true.mean():.1%}")

# ----- evaluation
try:
    y_pred = model.predict(X)
    y_score = get_scores(model, X)
except Exception as exc:
    st.error(f"The model could not score this data. Check the column values and types.\n\nDetails: {exc}")
    st.stop()

metrics = compute_metrics(y_true, y_pred, y_score)
st.subheader(f"Results - {model_name}")
for col, (name, value) in zip(st.columns(6), metrics.items()):
    col.metric(name, "n/a" if np.isnan(value) else f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.pyplot(plot_confusion(y_true, y_pred))
with right:
    st.markdown("**Classification report**")
    report = pd.DataFrame(classification_report(y_true, y_pred, labels=[0, 1], target_names=LABELS,
                                                output_dict=True, zero_division=0)).T
    report.loc["accuracy", "support"] = len(y_true)
    report["support"] = report["support"].astype(int)
    st.dataframe(report.round(4), width="stretch")

# ----- optional comparison
with st.expander("Compare all five models on this data"):
    if st.button("Run comparison"):
        rows = []
        for name, file in MODEL_FILES.items():
            m = load_model(file)
            rows.append({"Model": name, **compute_metrics(y_true, m.predict(X), get_scores(m, X))})
        comp = pd.DataFrame(rows).set_index("Model")
        st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#c6f6d5"), width="stretch")
