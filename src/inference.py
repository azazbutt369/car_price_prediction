import joblib
import pandas as pd

def predict(model_path, input_data: pd.DataFrame):
    model = joblib.load(model_path)
    preds = model.predict(input_data)
    return preds
