import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = df.copy()
    
    # Drop unnecessary columns
    df_processed = df_processed.drop(columns=['reg_code'], errors='ignore')
    
    # Drop rows with missing values in key columns
    df_processed = df_processed.dropna(subset=['mileage', 'standard_colour', 'body_type', 'fuel_type', 'year_of_registration'])
    
    # Feature engineering: car age
    current_year = pd.Timestamp('now').year
    df_processed['car_age'] = current_year - df_processed['year_of_registration']
    
    # Log transformation of price
    df_processed['log_price'] = df_processed['price'].apply(lambda x: max(x, 1)).apply(lambda x: np.log(x))
    
    return df_processed
