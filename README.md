# 📊 Home Credit Default Risk Prediction

An end-to-end Machine Learning pipeline built to evaluate loan default risk for credit applicants. The project handles high class imbalance and complex feature interactions to achieve top-tier prediction performance.

---

##  Performance Summary

| Phase | Model / Approach | Evaluation Metric (ROC-AUC) | Key Actions Taken |
| :--- | :--- | :---: | :--- |
| **Phase 1** | Baseline LightGBM | **0.74889** | Data preprocessing, SimpleImputer (median), Label & One-Hot Encoding. |
| **Phase 2** | LightGBM + Feature Engineering | **0.75760** | Engineered `CREDIT_TERM`, `CREDIT_INCOME_PERCENT`, and `EXT_SOURCES_*`. |
| **Phase 3** | CatBoost Classifier | **0.75897** | Applied CatBoost with depth=6 & learning_rate=0.04. |
| **Phase 4** | **Ensemble Model (Final)** | **`0.75991`** | Blended LightGBM (50%) + CatBoost (50%) for optimal generalization. |

- **Public Leaderboard Score:** **`0.7628`** 

---

##  Top Risk Drivers (Feature Importance)
1. **`CREDIT_TERM`**: Annuity to credit ratio was the single most decisive factor for default risk.
2. **`EXT_SOURCES_PROD` & `EXT_SOURCES_MEAN`**: Aggregated external credit scores provided strong predictive power.
3. **`DAYS_BIRTH` & `DAYS_EMPLOYED`**: Age and employment stability indicators highly influenced repayment capability.

---

## 🛠️ Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, LightGBM, CatBoost, Matplotlib, Seaborn

---

## 📁 Repository Contents
- `Home_Credit_Default_Risk.ipynb`: Full training & evaluation pipeline notebook.
- `final_ensemble_submission.csv`: Final ensemble model predictions.
