
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_cleaned.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def run_eda():

    # Load cleaned data
    df = pd.read_csv(DATA_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate churn percentage
    churn_rate = (df["Churn"] == "Yes").mean() * 100

    print("Total customers:", len(df))
    print("Churn rate:", round(churn_rate, 2), "%")

    # Churn distribution chart
    churn_counts = df["Churn"].value_counts()

    plt.figure(figsize=(8, 5))
    churn_counts.plot(kind="bar")

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn Status")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "churn_distribution.png")
    plt.show()

    # Churn by contract type
    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100

    print("\nChurn percentage by contract type:")
    print(contract_churn.round(2))

    contract_churn.plot(kind="bar", figsize=(9, 5))

    plt.title("Customer Churn by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Percentage of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "churn_by_contract.png")
    plt.show()


if __name__ == "__main__":
    run_eda()
