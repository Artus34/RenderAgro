import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="Crop Yield Prediction API",
    description="An API to predict crop yield with dynamic category loading.",
    version="2.0.0"
)

# --- 2. Load Saved Artifacts ---
# This is the same as before. All .pkl files are loaded at startup.
try:
    artifacts_path = "."
    model_path = os.path.join(artifacts_path, 'Best_Crop_Yield_Prediction_Model.pkl')
    scaler_path = os.path.join(artifacts_path, 'MinMaxScaler_Yield.pkl')
    columns_path = os.path.join(artifacts_path, 'X_Yield_Feature_Columns.pkl')
    mappings_path = os.path.join(artifacts_path, 'category_mappings.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(columns_path, 'rb') as f:
        model_columns = pickle.load(f)
    with open(mappings_path, 'rb') as f:
        # This .pkl file contains the simple ID-to-string mappings
        category_maps = pickle.load(f)

    # We also need the original dataframe to create the state->district dependency
    df = pd.read_csv('India Agriculture Crop Production.csv')

except FileNotFoundError as e:
    raise RuntimeError(f"Could not load a necessary artifact: {e}. Ensure all .pkl files and the .csv are in your GitHub repository.")


# --- 3. Define API Input/Output Models ---
class CropInput(BaseModel):
    state_id: int
    district_id: int
    crop_id: int
    season_id: int
    year: int
    area: float

class CategoryItem(BaseModel):
    id: int
    name: str

# --- 4. NEW: Dynamic Category Endpoints ---

@app.get("/categories", summary="Get Initial Categories", tags=["Categories"])
def get_initial_categories():
    """
    Provides the lists for States, Crops, and Seasons.
    Your Flutter app should call this once on startup.
    """
    return {
        "states": [{"id": k, "name": v} for k, v in category_maps['State'].items()],
        "crops": [{"id": k, "name": v} for k, v in category_maps['Crop'].items()],
        "seasons": [{"id": k, "name": v} for k, v in category_maps['Season'].items()]
    }

@app.get("/districts/{state_id}", summary="Get Districts for a State", response_model=list[CategoryItem], tags=["Categories"])
def get_districts_for_state(state_id: int):
    """
    Takes a state_id and returns a list of districts belonging to that state.
    Call this after the user selects a state.
    """
    # First, find the string name of the state from its ID
    state_name = category_maps['State'].get(state_id)
    if not state_name:
        raise HTTPException(status_code=404, detail="State ID not found.")

    # Filter the original dataframe to get unique districts for that state
    districts = sorted(df[df['State'] == state_name]['District'].unique())
    
    # Get the master ID for each district name
    district_master_map = {v: k for k, v in category_maps['District'].items()}
    
    response = []
    for district in districts:
        district_id = district_master_map.get(district)
        if district_id is not None:
            response.append({"id": district_id, "name": district})
            
    return response


# --- 5. Prediction Endpoint (Unchanged) ---
# This endpoint works exactly as before.
@app.post("/predict", summary="Predict Crop Yield", tags=["Prediction"])
def predict_yield(data: CropInput):
    """
    Predicts crop yield based on integer IDs for all features.
    """
    try:
        # Translate IDs to string values
        string_input = {
            'State': category_maps['State'].get(data.state_id),
            'District': category_maps['District'].get(data.district_id),
            'Crop': category_maps['Crop'].get(data.crop_id),
            'Season': category_maps['Season'].get(data.season_id)
        }
        for key, value in string_input.items():
            if value is None:
                raise HTTPException(status_code=400, detail=f"Invalid ID for '{key}'.")

        # Prepare DataFrame for prediction
        prediction_data = {**string_input, 'Area': data.area, 'StartYear': data.year}
        input_df = pd.DataFrame([prediction_data])
        input_encoded = pd.get_dummies(input_df, columns=['State', 'District', 'Crop', 'Season'])
        input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
        input_aligned[['Area', 'StartYear']] = scaler.transform(input_aligned[['Area', 'StartYear']])

        # Make prediction
        prediction = model.predict(input_aligned)
        final_prediction = max(0.0, prediction[0])

        return {"predicted_yield": f"{final_prediction:.2f}"}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

