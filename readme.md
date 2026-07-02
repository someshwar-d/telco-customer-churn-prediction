# Telco Customer Churn Prediction

A machine learning project to predict customer churn for a telecommunications company using the IBM Telco Customer Churn dataset.

---

## Project Overview

Customer churn — when a customer stops using a company's service — is one of the most critical challenges in the telecom industry. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project builds and compares multiple machine learning models to predict which customers are most likely to churn, enabling businesses to take proactive retention action.

---

##  Dataset

- **Source**: IBM Telco Customer Churn Dataset (Extended 33-column variant)
- **Records**: 7,043 customers
- **Features**: 33 columns including demographic info, services subscribed, account details, and churn status
- **Target Variable**: `Churn Value` (0 = Retained, 1 = Churned)

Key columns include:
- Customer demographics: Gender, Senior Citizen, Partner, Dependents
- Services: Phone Service, Internet Service, Online Security, Tech Support, Streaming TV/Movies
- Account info: Contract Type, Payment Method, Monthly Charges, Total Charges, Tenure Months
- Churn indicators: Churn Label, Churn Value, Churn Score, Churn Reason

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.13 | Core programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Machine learning models and preprocessing |
| Imbalanced-learn (SMOTE) | Handling class imbalance |
| XGBoost | Gradient boosting model |
| Jupyter Notebook | Development environment |

---

## Project Pipeline

### 1. Data Cleaning
- Fixed `Total Charges` column (blank strings → 0 for zero-tenure customers)
- Filled missing `Churn Reason` values with "Not Churned"
- Removed constant columns: `Count`, `Country`, `State`
- Confirmed zero duplicates across 7,043 records

### 2. Exploratory Data Analysis (EDA)
Key findings:
- **26.5% churn rate** vs 73.5% retained — class imbalance identified
- **Month-to-month contracts** churn far more than annual contracts
- **New customers** (low tenure) are the highest-risk group
- **Fiber optic users** churn at ~43%, nearly double DSL users (~18%)
- **Electronic check payers** churn significantly more than automatic payment users
- Customers with **higher monthly charges** are more likely to churn

### 3. Feature Engineering
- Dropped leakage columns: `Churn Score`, `Churn Reason`, `Churn Label`
- Dropped high-cardinality geographic columns: `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`
- Binary encoding for Yes/No columns (Gender, Partner, Dependents, etc.)
- One-hot encoding for multi-category columns (Contract, Internet Service, Payment Method, etc.)
- Final feature set: **31 features**

### 4. Modelling
- Train/test split: **80/20** with stratification
- Applied **SMOTE** to balance training data (4,139 samples per class)
- Applied **StandardScaler** for Logistic Regression
- Trained three models: Logistic Regression, Random Forest, XGBoost

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 78.4% | 58.1% | **67.4%** | **62.4%** |
| Random Forest | **78.8%** | **59.7%** | 61.5% | 60.6% |
| XGBoost | 78.7% | 59.5% | 62.0% | 60.7% |

> **Best model for churn use case**: Logistic Regression — highest Recall (67.4%) and F1 Score, meaning it catches the most actual churners, which is most valuable for business retention strategy.

---

##  Top 10 Features by Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Total Charges | 10.5% |
| 2 | Monthly Charges | 10.2% |
| 3 | Tenure Months | 10.1% |
| 4 | CLTV | 9.0% |
| 5 | Dependents | 6.9% |
| 6 | Contract_Two year | 6.3% |
| 7 | Online Security_Yes | 5.7% |
| 8 | Tech Support_Yes | 5.1% |
| 9 | Contract_One year | 3.8% |
| 10 | Online Backup_Yes | 2.9% |

---

##  Key Business Insights

- Customers on **month-to-month contracts** are the highest-risk group — offering incentives to switch to annual plans could significantly reduce churn
- **New customers in their first few months** are most vulnerable — targeted onboarding and early engagement programs are recommended
- Customers **without Online Security or Tech Support** churn more — bundling these services could improve retention
- **Electronic check** users churn at much higher rates — encouraging auto-payment adoption may help
- **Fiber optic** customers have unexpectedly high churn — pricing or service quality issues may need investigation

---

## 📁 Project Structure

```
telco-customer-churn-prediction/
│
├── churn_analysis.ipynb       # Main Jupyter notebook (full pipeline)
├── README.md                  # Project documentation
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset (not uploaded, see source)
```

---

##  How to Run

1. Clone this repository:
```bash
git clone https://github.com/someshwar-d/telco-customer-churn-prediction.git
```

2. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the `data/` folder

4. Launch Jupyter Notebook:
```bash
python -m notebook
```

5. Open `churn_analysis.ipynb` and run all cells

---

##  Author

**Someshwar D**  
Final Year B.Tech CSE Student  
Dhanalakshmi Srinivasan University, Tamil Nadu  
GitHub: [@someshwar-d](https://github.com/someshwar-d)
---
<linkedin
href="https://www.linkedin.com/in/(https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin)](www.linkedin.com/in/somesh-dhina-873b373aa)">

---

##  License

This project is open source and available under the [MIT License](LICENSE).
