
import pandas as pd
from pathlib import Path

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_cleaned.csv"
)


def analyze_customers():

    # Load cleaned dataset
    df = pd.read_csv(DATA_FILE)

    print("\n========== CUSTOMER INTELLIGENCE REPORT ==========")

    # 1. Churn by monthly charges
    print("\n1. CHURN BY MONTHLY CHARGES")

    charges = df.groupby("Churn")["MonthlyCharges"].mean()
    print(charges.round(2))

    # 2. Churn by customer tenure
    print("\n2. AVERAGE TENURE BY CHURN STATUS")

    tenure = df.groupby("Churn")["tenure"].mean()
    print(tenure.round(2))

    # 3. Churn by internet service
    print("\n3. CHURN BY INTERNET SERVICE")

    internet = pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100

    print(internet.round(2))

    # 4. Churn by payment method
    print("\n4. CHURN BY PAYMENT METHOD")

    payment = pd.crosstab(
        df["PaymentMethod"],
        df["Churn"],
        normalize="index"
    ) * 100

    print(payment.round(2))

    # 5. Churn by senior citizen status
    print("\n5. CHURN BY SENIOR CITIZEN STATUS")

    senior = pd.crosstab(
        df["SeniorCitizen"],
        df["Churn"],
        normalize="index"
    ) * 100

    print(senior.round(2))


if __name__ == "__main__":
    analyze_customers()
