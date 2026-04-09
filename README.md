# keepify-churn-model-prediction
 Keepify — predict which telecom customers are about to leave, before they do. End-to-end churn prediction using Python &amp; scikit-learn on the IBM Telco dataset.

# 🔒 Keepify — Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square)
![Pandas](https://img.shields.io/badge/pandas-1.5%2B-150458?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> **Keep what matters. Drop what doesn't.**
> Keepify predicts which telecom customers are about to churn — before they do.

---

## 📌 Overview

Customer churn is one of the most expensive problems in the telecom industry. Acquiring a new customer costs 5–7x more than retaining an existing one.

**Keepify** is an end-to-end machine learning pipeline that classifies customers as likely to churn or stay, using the IBM Telco Customer Churn dataset. It trains two models — Logistic Regression and Random Forest — and provides actionable feature insights so businesses know exactly *what* drives customers away.

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | IBM Telco Customer Churn (public) |
| Rows | 7,043 customers |
| Features | 20 (demographics, services, billing) |
| Target | `Churn` — Yes / No |
| Class balance | ~73% No Churn / ~27% Churn |

**Key features include:** tenure, contract type, monthly charges, total charges, internet service, payment method, and more.

---

## 🗂️ Project Structure

```
keepify/
├── keepify_model.py          # Main ML pipeline (all steps)
├── requirements.txt          # Dependencies
├── README.md                 # You are here
├── .gitignore
└── outputs/
    ├── confusion_matrix.png  # Model evaluation plot
    └── feature_importance.png # Top churn drivers
```

---

## ⚙️ Workflow

```
Raw Data → Clean → Encode → Scale → Split → Train → Evaluate → Visualize
```

1. **Load** — Fetch dataset from public IBM GitHub URL
2. **Clean** — Convert `TotalCharges` to numeric, fill nulls, drop `customerID`
3. **Encode** — Label encode all categorical columns
4. **Scale** — Standardize features with `StandardScaler`
5. **Split** — 80% train / 20% test (random state = 42)
6. **Train** — Logistic Regression + Random Forest (100 estimators)
7. **Evaluate** — Accuracy, precision, recall, F1-score, confusion matrix
8. **Visualize** — Feature importance bar chart

---

## 🤖 Models & Results

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | ~80% | Fast, interpretable baseline |
| Random Forest | ~79–82% | Stronger on non-linear patterns |

> Exact metrics vary slightly by run. Use `random_state=42` to reproduce.

**Top churn drivers (from Random Forest):**
- Monthly Charges
- Tenure
- Total Charges
- Contract Type
- Internet Service

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/keepify.git
cd keepify
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the model

```bash
python keepify_model.py
```

Outputs will be printed to the console and plots will render inline (or save to `outputs/` if configured).

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |

---

## 📁 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 📈 Sample Output

**Confusion Matrix** — Random Forest predictions vs actual labels
**Feature Importance** — Which features most influence churn

*(Run the script to generate these plots)*

---

## 🙋 Use Cases

- Telecom companies wanting to reduce churn rate
- Data science portfolio project demonstrating end-to-end ML
- Learning classification, encoding, scaling, and evaluation in Python

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

<p align="center">Built with Python · Powered by scikit-learn · Inspired by the cost of losing customers</p>
