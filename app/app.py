import streamlit as st
import requests
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Intelligence Platform")
st.subheader("AI-Powered Customer Churn Prediction")

st.write("Enter customer information to predict churn risk.")

API_URL = "https://customer-intelligence-api-dgxl.onrender.com/prediction/churn"

# =====================================
# EXECUTIVE CUSTOMER INTELLIGENCE DASHBOARD
# =====================================

st.divider()

st.header("Executive Dashboard")
st.write("Overview of customer behaviour and churn performance.")

# Load cleaned customer dataset
@st.cache_data
def load_customer_data():
    return pd.read_csv(
        "data/processed/customer_churn_cleaned.csv"
    )

try:
    df = load_customer_data()

    total_customers = len(df)
    churned_customers = (df["Churn"] == "Yes").sum()
    churn_rate = (churned_customers / total_customers) * 100
    average_monthly_charges = df["MonthlyCharges"].mean()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Historical Churn Rate",
        f"{churn_rate:.2f}%"
    )

    col4.metric(
        "Average Monthly Charges",
        f"{average_monthly_charges:.2f}"
    )

    st.divider()

    # Churn distribution
    st.subheader("Customer Churn Distribution")

    churn_counts = df["Churn"].value_counts()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("Churned vs Retained Customers")
        st.bar_chart(churn_counts)

    with chart_col2:
        st.write("Churn by Contract Type")

        contract_churn = pd.crosstab(
            df["Contract"],
            df["Churn"]
        )

        st.bar_chart(contract_churn)

except Exception as error:
    st.error(f"Could not load dashboard data: {error}")

# =====================================
# CUSTOMER RISK SEGMENTATION
# =====================================

st.divider()
st.header("Customer Risk Segmentation")

# Estimate customer risk using historical churn rates
risk_df = df.copy()

risk_df["Risk Segment"] = pd.cut(
    risk_df["tenure"],
    bins=[-1, 12, 36, float("inf")],
    labels=["High Risk", "Medium Risk", "Low Risk"]
)

risk_counts = risk_df["Risk Segment"].value_counts()

col1, col2, col3 = st.columns(3)

col1.metric("High Risk Customers", int(risk_counts.get("High Risk", 0)))
col2.metric("Medium Risk Customers", int(risk_counts.get("Medium Risk", 0)))
col3.metric("Low Risk Customers", int(risk_counts.get("Low Risk", 0)))

st.subheader("Risk Segment Distribution")
st.bar_chart(risk_counts)

# Actual historical churn rate by segment
segment_churn = (
    risk_df.groupby("Risk Segment", observed=False)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

st.subheader("Historical Churn Rate by Risk Segment")
st.bar_chart(segment_churn)



# =====================================
# ML-BASED CUSTOMER RISK SEGMENTATION
# =====================================

st.divider()
st.header("AI-Based Customer Risk Segmentation")

MODEL_PATH = "models/logistic_regression.joblib"

@st.cache_resource
def load_churn_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_churn_model()

    # Remove target column before prediction
    customer_features = df.drop(columns=["Churn"])

    # Predict churn probability for every customer
    churn_probabilities = model.predict_proba(
        customer_features
    )[:, 1]

    risk_df = df.copy()
    risk_df["Churn Probability"] = churn_probabilities

    # Assign model-based risk categories
    risk_df["Risk Segment"] = pd.cut(
        risk_df["Churn Probability"],
        bins=[-0.001, 0.40, 0.70, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    risk_counts = risk_df["Risk Segment"].value_counts()

    # Display risk KPIs
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "High Risk Customers",
        int(risk_counts.get("High Risk", 0))
    )

    col2.metric(
        "Medium Risk Customers",
        int(risk_counts.get("Medium Risk", 0))
    )

    col3.metric(
        "Low Risk Customers",
        int(risk_counts.get("Low Risk", 0))
    )

    st.subheader("AI-Predicted Customer Risk Distribution")
    st.bar_chart(risk_counts)

    st.subheader("Customer Risk Details")

    risk_table = risk_df[
        ["tenure", "MonthlyCharges", "Churn Probability", "Risk Segment"]
    ].copy()

    risk_table["Churn Probability"] = (
        risk_table["Churn Probability"] * 100
    ).round(2)

    risk_table = risk_table.rename(columns={
        "tenure": "Tenure (Months)",
        "MonthlyCharges": "Monthly Charges",
        "Churn Probability": "Churn Probability (%)"
    })

    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Risk categories use model-estimated churn probabilities. "
        "Low: below 40%; Medium: 40% to below 70%; "
        "High: 70% or above. These are provisional business thresholds."
    )

    # REVENUE AT RISK ANALYSIS

    st.divider()
    st.header("Revenue at Risk Analysis")

    high_risk_customers = risk_df[
        risk_df["Risk Segment"] == "High Risk"
    ]

    high_risk_count = len(high_risk_customers)

    monthly_revenue_at_risk = high_risk_customers[
        "MonthlyCharges"
    ].sum()

    annual_revenue_at_risk = monthly_revenue_at_risk * 12

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "High-Risk Customers",
        f"{high_risk_count:,}"
    )

    col2.metric(
        "Monthly Revenue at Risk",
        f"{monthly_revenue_at_risk:,.2f}"
    )

    col3.metric(
        "Annualized Revenue Exposure",
        f"{annual_revenue_at_risk:,.2f}"
    )

    st.subheader("Revenue at Risk by Segment")

    revenue_summary = risk_df.groupby(
        "Risk Segment", observed=False
    )["MonthlyCharges"].sum()

    st.bar_chart(revenue_summary)

    st.caption(
        "Revenue exposure is based on monthly charges of customers "
        "classified as high risk. Annualized exposure is an estimate, "
        "not a forecast of actual revenue loss."
    )

except Exception as error:
    st.error(f"Customer risk analysis failed: {error}")



# Customer information form
with st.form("customer_form"):

    st.header("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", 0, 100, 2)

    with col2:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.header("Billing Information")

    bill1, bill2 = st.columns(2)

    with bill1:
        monthly = st.number_input(
            "Monthly Charges", min_value=0.0, value=85.5
        )

    with bill2:
        total = st.number_input(
            "Total Charges", min_value=0.0, value=171.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

if submitted:

    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    try:
        response = requests.post(API_URL, json=customer_data, timeout=15)
        response.raise_for_status()

        result = response.json()

        st.header("Churn Prediction Result")

        probability = result["churn_probability"]
        risk = result["risk_category"]

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(probability)

        if risk == "High":
            st.error(f"High Churn Risk: {risk}")

        elif risk == "Medium":
            st.warning(f"Medium Churn Risk: {risk}")

        else:
            st.success(f"Low Churn Risk: {risk}")

        st.write(result["message"])

    except requests.exceptions.RequestException as error:
        st.error(f"Could not connect to the prediction API: {error}")
