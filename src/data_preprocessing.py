
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "customer_churn_cleaned.csv"


def preprocess_data():
    # Load dataset
    df = pd.read_csv(RAW_FILE)

    print("Original dataset shape:", df.shape)

    # Convert TotalCharges into numeric format
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Replace missing TotalCharges with zero
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Remove duplicate records
    df = df.drop_duplicates()

    # Create processed folder
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save cleaned dataset
    df.to_csv(PROCESSED_FILE, index=False)

    print("Cleaned dataset shape:", df.shape)
    print("Remaining missing values:", df.isnull().sum().sum())
    print("Remaining duplicates:", df.duplicated().sum())
    print("Saved file:", PROCESSED_FILE)


if __name__ == "__main__":
    preprocess_data()
