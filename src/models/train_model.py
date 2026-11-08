
import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.pipeline import Pipeline

from src.pipelines.preprocessing import build_preprocessor


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_models():

    # Load training and testing datasets
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(X_train)

    # Define machine learning algorithms
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    results = []

    # Train and evaluate models
    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, probabilities)
        }

        results.append(metrics)

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0
        ))

        # Save trained model
        filename = model_name.lower().replace(" ", "_")

        joblib.dump(
            pipeline,
            MODEL_DIR / f"{filename}.joblib"
        )

    # Compare model performance
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False
    )

    print("\n========== MODEL COMPARISON ==========")
    print(results_df.round(4).to_string(index=False))

    print("\nModels saved successfully!")


if __name__ == "__main__":
    train_models()
