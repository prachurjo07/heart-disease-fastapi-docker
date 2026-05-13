
from pydantic import BaseModel, Field


class HeartDiseaseInput(BaseModel):
    age: int = Field(..., example=52)
    sex: int = Field(..., example=1)
    cp: int = Field(..., example=0)
    trestbps: int = Field(..., example=125)
    chol: int = Field(..., example=212)
    fbs: int = Field(..., example=0)
    restecg: int = Field(..., example=1)
    thalach: int = Field(..., example=168)
    exang: int = Field(..., example=0)
    oldpeak: float = Field(..., example=1.0)
    slope: int = Field(..., example=2)
    ca: int = Field(..., example=2)
    thal: int = Field(..., example=3)