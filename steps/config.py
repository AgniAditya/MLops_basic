from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """Model Configs"""
    model_name: str = "LinearRegression"