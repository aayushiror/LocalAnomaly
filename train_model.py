"""
Train ML models for Network Anomaly Detection on NSL-KDD dataset.

Models:
  1. Random Forest         — best supervised accuracy
  2. Gradient Boosting     — sklearn ensemble, strong baseline
  3. Isolation Forest      — unsupervised (trains on normal traffic only)

Artifacts saved to models/:
  preprocessor.pkl, rf_model.pkl, gb_model.pkl, iso_forest.pkl,
  feature_names.pkl, label_encoder.pkl, feature_importance.pkl, metrics.json
"""

import os, json, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, IsolationForest
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, roc_auc_score,
)

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
CAT_COLS   = ["protocol_type", "service", "flag"]


def load_data():
    return (pd.read_csv(os.path.join(DATA_DIR, "train.csv")),
            pd.read_csv(os.path.join(DATA_DIR, "test.csv")))


def build_preprocessor(X):
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("num", StandardScaler(), num_cols),
    ])
    return pre.fit(X)


def feature_names_out(pre, X):
    cats = pre.named_transformers_["cat"].get_feature_names_out(CAT_COLS).tolist()
    nums = [c for c in X.columns if c not in CAT_COLS]
    return cats + nums


def save_pkl(obj, name):
    with open(os.path.join(MODELS_DIR, name), "wb") as f:
        pickle.dump(obj, f)
    print(f"  {name}")


def evaluate(name, y_true, y_pred, y_prob=None):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    print(f"\n{'─'*52}")
    print(f"  {name}")
    print(f"{'─'*52}")
    print(f"  Accuracy  {acc:.4f}   F1 {f1:.4f}   Precision {prec:.4f}   Recall {rec:.4f}")
    if auc:
        print(f"  ROC-AUC   {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))
    return dict(accuracy=round(acc,4), f1=round(f1,4), precision=round(prec,4),
                recall=round(rec,4), auc=round(auc,4) if auc else None,
                confusion_matrix=confusion_matrix(y_true, y_pred).tolist())


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Loading data …")
    train_df, test_df = load_data()

    feat_cols = [c for c in train_df.columns if c not in ["binary_label","attack_type"]]
    X_tr, y_tr = train_df[feat_cols], train_df["binary_label"]
    X_te, y_te = test_df[feat_cols],  test_df["binary_label"]
    print(f"Train {X_tr.shape}  |  Test {X_te.shape}")
    print(f"Normal/Attack  train: {(y_tr==0).sum()}/{(y_tr==1).sum()}")

    le = LabelEncoder().fit(train_df["attack_type"])
    save_pkl(le, "label_encoder.pkl")

    print("\nPreprocessing …")
    pre   = build_preprocessor(X_tr)
    fnames = feature_names_out(pre, X_tr)
    Xtr_t = pre.transform(X_tr)
    Xte_t = pre.transform(X_te)
    save_pkl(pre,    "preprocessor.pkl")
    save_pkl(fnames, "feature_names.pkl")

    metrics = {}

    # 1. Random Forest
    print("\n[1/3] Random Forest …")
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5,
                                 n_jobs=-1, random_state=42, class_weight="balanced")
    rf.fit(Xtr_t, y_tr)
    metrics["random_forest"] = evaluate("Random Forest", y_te,
                                         rf.predict(Xte_t), rf.predict_proba(Xte_t)[:,1])
    save_pkl(rf, "rf_model.pkl")

    # 2. Gradient Boosting (subsample for speed)
    print("\n[2/3] Gradient Boosting (40k subsample) …")
    Xs, ys = resample(Xtr_t, y_tr, n_samples=40_000, stratify=y_tr, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=6, learning_rate=0.15,
                                     subsample=0.8, random_state=42)
    gb.fit(Xs, ys)
    metrics["gradient_boosting"] = evaluate("Gradient Boosting", y_te,
                                             gb.predict(Xte_t), gb.predict_proba(Xte_t)[:,1])
    save_pkl(gb, "gb_model.pkl")

    # 3. Isolation Forest
    print("\n[3/3] Isolation Forest (unsupervised) …")
    iso = IsolationForest(n_estimators=200, contamination=0.3, random_state=42, n_jobs=-1)
    iso.fit(Xtr_t[y_tr == 0])
    iso_pred = (iso.predict(Xte_t) == -1).astype(int)
    metrics["isolation_forest"] = evaluate("Isolation Forest", y_te, iso_pred)
    save_pkl(iso, "iso_forest.pkl")

    ###featureimportance
    fi = pd.Series(rf.feature_importances_, index=fnames).sort_values(ascending=False)
    print("\nTop 15 features (Random Forest):")
    for feat, imp in fi.head(15).items():
        print(f"  {feat:<42} {imp:.4f}")
    save_pkl(fi.head(30).to_dict(), "feature_importance.pkl")

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTraining complete!  Run: streamlit run app/app.py")


if __name__ == "__main__":
    main()
