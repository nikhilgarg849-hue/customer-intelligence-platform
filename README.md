# Customer Intelligence & Churn Prediction Platform

An end-to-end machine learning platform designed to analyze customer behavior, predict customer churn, identify high-risk customers, and support data-driven retention strategies.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. This project uses machine learning and customer analytics to identify customers who are more likely to discontinue their services.

The platform combines data preprocessing, exploratory data analysis, machine learning, model explainability, a FastAPI prediction backend, and an interactive Streamlit dashboard.

## Key Features

- Customer data analysis and exploratory data analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering and model pipelines
- Customer churn prediction
- Logistic Regression and Random Forest models
- Model evaluation and comparison
- SHAP-based model explainability
- Customer risk segmentation
- Revenue-at-risk analysis
- Individual customer churn prediction
- REST API using FastAPI
- Interactive analytics dashboard using Streamlit

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| API Development | FastAPI |
| Dashboard | Streamlit |
| Model Storage | Joblib |
| Version Control | Git, GitHub |

## Machine Learning Models

Two classification models were developed and evaluated:

1. Logistic Regression
2. Random Forest Classifier

### Model Performance

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 73.81% | 75.87% |
| Precision | 50.43% | 53.22% |
| Recall | 78.34% | 75.13% |
| F1 Score | 61.36% | 62.31% |
| ROC-AUC | 84.16% | 83.78% |

Random Forest achieved better accuracy, precision, and F1 score, while Logistic Regression achieved higher recall and ROC-AUC.

The appropriate model depends on the business objective and the cost of missing customers who may churn.

## Explainable AI

SHAP was used to interpret the Random Forest model and understand the features influencing churn predictions.

Important features identified include:

- Month-to-month contracts
- Customer tenure
- Fiber optic internet service
- Two-year contracts
- Online security
- Total charges
- Technical support
- Payment method

SHAP feature importance indicates contribution magnitude, not necessarily whether a feature increases or decreases churn risk.

## Application Architecture

```text
Customer Dataset
       |
       v
Data Preprocessing
       |
       v
Exploratory Data Analysis
       |
       v
Feature Engineering
       |
       v
Machine Learning Models
       |
       +----------------------+
       |                      |
       v                      v
Model Evaluation         SHAP Explainability
       |
       v
Saved ML Model
       |
       v
FastAPI Prediction Backend
       |
       v
Streamlit Dashboard
       |
       v
Customer Insights & Risk Analysis
```

## Project Structure

```text
customer-intelligence-platform/
│
├── api/
│   ├── main.py
│   └── prediction.py
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── model_comparison.csv
│   └── shap_feature_importance.csv
│
├── reports/
│   └── figures/
│
├── src/
│   ├── customer_analysis.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── models/
│   └── pipelines/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The project uses the Telco Customer Churn dataset.

- Total records: 7,043
- Total columns: 21
- Target variable: Churn
- Classification problem: Binary classification

The dataset contains customer demographics, service subscriptions, contract information, payment methods, tenure, and billing information.

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/nikhilgarg849-hue/customer-intelligence-platform.git
cd customer-intelligence-platform
```

### 2. Create and activate the Conda environment

```bash
conda create -n customer-intelligence python=3.11 -y
conda activate customer-intelligence
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend

Open Terminal 1:

```bash
uvicorn api.main:app --reload --port 8000
```

API documentation:

http://127.0.0.1:8000/docs

### 5. Start the Streamlit dashboard

Open Terminal 2:

```bash
streamlit run app/app.py
```

The dashboard will be available at:

http://localhost:8501

Keep both services running while using the local application.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/prediction/churn` | Predict customer churn |

## Business Value

The platform demonstrates how machine learning can support business decisions through:

- Early identification of customers at risk of leaving
- Customer risk segmentation
- Identification of churn-related patterns
- Revenue exposure analysis
- Explainable predictions to support retention planning

Revenue-at-risk figures are analytical estimates based on the selected customer segment, not guaranteed future revenue losses.

## Future Improvements

- Cloud deployment of the Streamlit application and FastAPI backend
- PostgreSQL database integration
- Automated model retraining
- Customer retention recommendation engine
- Model monitoring and drift detection
- Authentication and role-based access control
- Docker-based deployment

## Author

**Nikhil Garg**

MBA – Healthcare Analytics  
IIHMR University, Jaipur

GitHub: https://github.com/nikhilgarg849-hue

---

*This project is developed for learning, analytics, and demonstration of an end-to-end customer intelligence and machine learning workflow.*