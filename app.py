from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Initialize FastAPI
app = FastAPI()

# Load model, scaler, and feature columns
with open("Best_Crop_Yield_Prediction_Model.pkl", "rb") as f:
    yield_model = pickle.load(f)

with open("MinMaxScaler_Yield.pkl", "rb") as f:
    ms_yield = pickle.load(f)

with open("X_Yield_Feature_Columns.pkl", "rb") as f:
    X_yield_final_columns = pickle.load(f)

# Request schema (IDs instead of strings)
class YieldInput(BaseModel):
    state: int
    district: int
    crop: int
    season: int
    year: int
    area: float

# Endpoint
@app.post("/predict/yield")
def predict_yield(data: YieldInput):
    try:
        # Build dataframe with feature columns
        input_data = pd.DataFrame([[data.state, data.district, data.crop, data.season, data.year, data.area]],
                                  columns=["State_Name", "District_Name", "Crop", "Season", "StartYear", "Area"])

        # Scale only numeric continuous features
        input_data[["Area", "StartYear"]] = ms_yield.transform(input_data[["Area", "StartYear"]])

        # Reorder to match training feature columns
        input_data = input_data[X_yield_final_columns]

        # Predict
        prediction = yield_model.predict(input_data.values)[0]
        prediction = max(0.0, prediction)  # avoid negative values

        return {"predicted_yield": round(float(prediction), 2)}

    except Exception as e:
        return {"error": str(e)}
