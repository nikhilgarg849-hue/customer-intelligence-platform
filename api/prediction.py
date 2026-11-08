
import joblib
import pandas as pd

from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.joblib"

router = APIRouter(prefix="/prediction", tags=["Churn Prediction"])


# Customer input schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@router.post("/churn")
def predict_churn(customer: CustomerData):

    try:
        # Load trained model
        model = joblib.load(MODEL_PATH)

        # Convert customer input to DataFrame
        customer_df = pd.DataFrame(
            [customer.model_dump()]
        )

        # Generate prediction
        prediction = int(model.predict(customer_df)[0])

        # Get churn probability
        probability = float(
            model.predict_proba(customer_df)[0][1]
        )

        # Assign risk category
        if probability >= 0.70:
            risk = "High"
        elif probability >= 0.40:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "risk_category": risk,
            "message": (
                "Customer is likely to churn"
                if prediction == 1
                else "Customer is likely to stay"
            )
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Trained model file not found."
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
