import pandas as pd
from sklearn.preprocessing import StandardScaler

def engineer_features(data):
    """
    Perform feature engineering on the input data.
    
    Args:
    data (pandas.DataFrame): Input data
    
    Returns:
    pandas.DataFrame: Data with engineered features
    """
    # Create lag features
    data['demand_lag1'] = data['demand'].shift(1)
    data['demand_lag7'] = data['demand'].shift(7)
    
    # Create rolling mean features
    data['demand_rolling_mean_7'] = data['demand'].rolling(window=7).mean()
    data['demand_rolling_mean_30'] = data['demand'].rolling(window=30).mean()
    
    # Create interaction features
    data['temp_weekend_interaction'] = data['temperature'] * data['is_weekend']
    data['marketing_holiday_interaction'] = data['marketing_spend'] * data['is_holiday']
    
    # Drop rows with NaN values resulting from lag and rolling features
    data = data.dropna()
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = ['temperature', 'marketing_spend', 'demand_lag1', 'demand_lag7', 
                          'demand_rolling_mean_7', 'demand_rolling_mean_30']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    return data