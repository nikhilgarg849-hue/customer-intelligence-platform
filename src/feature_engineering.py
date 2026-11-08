
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_cleaned.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def create_features():

    # Load cleaned dataset
    df = pd.read_csv(DATA_FILE)

    print("Original dataset shape:", df.shape)

    # Remove customer ID (identifier, not a predictive feature)
    df = df.drop(columns=["customerID"])

    # Convert target variable into binary format
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Separate predictors and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Save the datasets
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    print("\nFeature engineering completed!")
    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)
    print("Training target:", y_train.shape)
    print("Testing target:", y_test.shape)

    print("\nTraining churn distribution:")
    print(y_train.value_counts(normalize=True).round(3))


if __name__ == "__main__":
    create_features()
