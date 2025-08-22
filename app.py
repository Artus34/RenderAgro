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

# Request schema
class YieldInput(BaseModel):
    State: str
    District: str
    Crop: str
    Season: str
    Year: int
    Area: float

# Helper function: build input row
def build_input(data: YieldInput):
    # Start with all 0s
    input_dict = {col: [0] for col in X_yield_final_columns}
    df_input = pd.DataFrame(input_dict)

    # Fill numeric features
    df_input["Area"] = data.Area
    df_input["StartYear"] = data.Year

    # One-hot encode categorical features
    def set_one_hot(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in X_yield_final_columns:
            df_input[col_name] = 1
        else:
            raise ValueError(f"Invalid {prefix}: '{value}' not in training data.")

    set_one_hot("State", data.State)
    set_one_hot("District", data.District)
    set_one_hot("Crop", data.Crop)
    set_one_hot("Season", data.Season)

    # Scale numerical values
    df_input[["Area", "StartYear"]] = ms_yield.transform(df_input[["Area", "StartYear"]])

    return df_input[X_yield_final_columns]

# Endpoint
@app.post("/predict_yield")
def predict_yield(data: YieldInput):
    try:
        df_input = build_input(data)
        prediction = yield_model.predict(df_input.values)[0]
        prediction = max(0.0, prediction)  # avoid negative
        return {"predicted_yield": round(float(prediction), 2)}
    except Exception as e:
        return {"error": str(e)}
