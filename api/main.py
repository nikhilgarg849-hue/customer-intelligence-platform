from api.prediction import router as prediction_router
from fastapi import FastAPI

app = FastAPI(
    title="Customer Intelligence Platform",
    description="Customer analytics and churn prediction API",
    version="1.0.0",
)

app.include_router(prediction_router)

@app.get("/")
def root():
    return {
        "message": "Customer Intelligence Platform API",
        "status": "running",
    }

@app.get("/health")
def health():
    return {"status": "healthy"}
