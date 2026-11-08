
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_models():

    # Load test data
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    model_files = {
        "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
        "Random Forest": MODEL_DIR / "random_forest.joblib"
    }

    for model_name, model_path in model_files.items():

        print(f"\n========== {model_name} ==========")

        # Load trained model
        model = joblib.load(model_path)

        # Generate predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0
        ))

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, predictions)

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Retained", "Churned"]
        )

        display.plot()
        plt.title(f"{model_name} - Confusion Matrix")
        plt.tight_layout()

        filename = model_name.lower().replace(" ", "_")
        plt.savefig(
            FIGURE_DIR / f"{filename}_confusion_matrix.png"
        )
        plt.close()

        # 2. ROC Curve
        fpr, tpr, thresholds = roc_curve(
            y_test,
            probabilities
        )

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 5))
        plt.plot(
            fpr,
            tpr,
            label=f"ROC-AUC = {roc_auc:.3f}"
        )
        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.title(f"{model_name} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / f"{filename}_roc_curve.png"
        )
        plt.close()

        # 3. Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(
            y_test,
            probabilities
        )

        plt.figure(figsize=(8, 5))
        plt.plot(recall, precision)

        plt.title(f"{model_name} - Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / f"{filename}_precision_recall.png"
        )
        plt.close()

        print("ROC-AUC:", round(roc_auc, 4))
        print("Evaluation charts saved successfully!")

    print("\nAll model evaluations completed!")


if __name__ == "__main__":
    evaluate_models()
