
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_preprocessor(X):

    # Identify numerical and categorical columns
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


if __name__ == "__main__":

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed"

    X_train = pd.read_csv(data_dir / "X_train.csv")

    preprocessor = build_preprocessor(X_train)

    transformed_data = preprocessor.fit_transform(X_train)

    print("Preprocessing pipeline created successfully!")
    print("Original features:", X_train.shape[1])
    print("Transformed features:", transformed_data.shape[1])
    print("Training records:", transformed_data.shape[0])
