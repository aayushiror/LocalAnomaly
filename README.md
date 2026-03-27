# local anomaly
A full end-to-end Machine Learning project for detecting network intrusions using the **NSL-KDD** dataset — the gold-standard benchmark dataset from the Canadian Institute for Cybersecurity.

---

## Project Structure

```
network_anomaly_detection/
├── data/                   # Auto-downloaded NSL-KDD dataset
│   ├── KDDTrain+.txt
│   ├── KDDTest+.txt
│   ├── train.csv           # preprocessed
│   └── test.csv
├── models/                 # Trained model artifacts
│   ├── preprocessor.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── iso_forest.pkl
│   ├── feature_names.pkl
│   ├── label_encoder.pkl
│   ├── feature_importance.pkl
│   └── metrics.json
├── scripts/
│   ├── download_data.py    # Downloads & preprocesses NSL-KDD
│   └── train_model.py      # Trains RF, XGBoost, Isolation Forest
├── app/
│   └── app.py              # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Dataset — NSL-KDD

| Property | Value |
|---|---|
| Source | Canadian Institute for Cybersecurity |
| Train samples | 125,973 |
| Test samples | 22,544 |
| Features | 41 network traffic features |
| Labels | normal, DoS, Probe, R2L, U2R |

**Why NSL-KDD?**
- Improved version of KDD Cup 1999 (no duplicate records)
- Industry-standard benchmark for IDS research
- Well-balanced train/test split

---

## Models

| Model | Type | Notes |
|---|---|---|
| **Random Forest** | Supervised | Best accuracy, fast inference |
| **XGBoost** | Supervised | Handles imbalanced data well |
| **Isolation Forest** | Unsupervised | No labels needed, trains on normal traffic only |

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
```bash
python scripts/download_data.py
```

### 3. Train models
```bash
python scripts/train_model.py
```
> Training takes ~2-5 minutes depending on hardware.

### 4. Launch the app
```bash
streamlit run app/app.py
```

---

##  Streamlit App Features

### Anomaly Detector
- Pick **random test samples** or filter by attack type
- **Real-time prediction** with all 3 models
- Threat score **gauge chart**
- Ground truth vs prediction comparison
- **Feature radar chart** for traffic visualization
- Full feature breakdown table

###  Model Performance
- Accuracy, F1, Precision, Recall for all models
- Side-by-side confusion matrices
- Feature importance bar chart

###  EDA & Insights
- Attack type distribution (pie chart)
- Protocol vs label breakdown
- Byte transfer distribution
- Error rate scatter plot

---

##  Expected Results

| Model | Accuracy | F1 |
|---|---|---|
| Random Forest | ~99.2% | ~0.993 |
| XGBoost | ~98.8% | ~0.989 |
| Isolation Forest | ~76–82% | ~0.77 |

---

##  Feature Engineering

Key features used by the model:
- `src_bytes`, `dst_bytes` — data transfer volume
- `count`, `srv_count` — connection frequency (2-second window)
- `serror_rate`, `rerror_rate` — SYN/REJ error rates
- `same_srv_rate`, `diff_srv_rate` — service diversity
- `protocol_type`, `service`, `flag` — categorical (one-hot encoded)
- `dst_host_*` — destination host statistics

---

##  References

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) — Canadian Institute for Cybersecurity
- Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set", CISDA 2009
- GitHub mirror: [jmnwong/NSL-KDD-Dataset](https://github.com/jmnwong/NSL-KDD-Dataset)
