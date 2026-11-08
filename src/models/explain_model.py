
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import sparse


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def explain_model():

    # Load test dataset
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    # Load trained Random Forest pipeline
    model = joblib.load(MODEL_DIR / "random_forest.joblib")

    print("Model loaded successfully!")

    # Separate preprocessing and classifier
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Transform test data
    X_transformed = preprocessor.transform(X_test)

    # Convert sparse matrix to dense if necessary
    if sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()

    # Select a sample to reduce computation time
    sample_size = min(300, len(X_transformed))

    X_sample = X_transformed[:sample_size]

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    X_sample_df = pd.DataFrame(
        X_sample,
        columns=feature_names
    )

    print("Calculating SHAP values...")

    # Create SHAP Tree Explainer
    explainer = shap.TreeExplainer(classifier)

    shap_values = explainer.shap_values(X_sample)

    # Extract churn class (class 1)
    if isinstance(shap_values, list):
        churn_values = shap_values[1]

    elif np.ndim(shap_values) == 3:
        churn_values = shap_values[:, :, 1]

    else:
        churn_values = shap_values

    # Get expected value for churn class
    base_value = explainer.expected_value

    if np.ndim(base_value) > 0:
        base_value = base_value[1]

    # Create SHAP explanation object
    explanation = shap.Explanation(
        values=churn_values,
        base_values=np.full(sample_size, base_value),
        data=X_sample_df.values,
        feature_names=feature_names
    )

    # 1. SHAP Summary Plot
    plt.figure()
    shap.summary_plot(
        churn_values,
        X_sample_df,
        show=False,
        max_display=15
    )

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "shap_summary.png",
        bbox_inches="tight"
    )
    plt.close()

    # 2. SHAP Feature Importance
    plt.figure()
    shap.summary_plot(
        churn_values,
        X_sample_df,
        plot_type="bar",
        show=False,
        max_display=15
    )

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "shap_feature_importance.png",
        bbox_inches="tight"
    )
    plt.close()

    # 3. Individual Customer Explanation
    plt.figure()
    shap.plots.waterfall(
        explanation[0],
        max_display=15,
        show=False
    )

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "individual_customer_explanation.png",
        bbox_inches="tight"
    )
    plt.close()

    # Print global feature importance
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_SHAP_Importance": np.abs(churn_values).mean(axis=0)
    })

    importance = importance.sort_values(
        by="Mean_SHAP_Importance",
        ascending=False
    )

    print("\nTOP 15 CHURN PREDICTION FEATURES")
    print(importance.head(15).to_string(index=False))

    importance.to_csv(
        MODEL_DIR / "shap_feature_importance.csv",
        index=False
    )

    print("\nSHAP analysis completed successfully!")
    print("Charts saved in:", FIGURE_DIR)


if __name__ == "__main__":
    explain_model()
