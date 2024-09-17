import pandas as pd
import numpy as np

def generate_synthetic_data(start_date='2023-01-01', end_date='2023-12-31'):
    """
    Generate synthetic data for food delivery demand prediction.
    
    Args:
    start_date (str): Start date for the data range
    end_date (str): End date for the data range
    
    Returns:
    pandas.DataFrame: Synthetic data for demand prediction
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_samples = len(dates)
    
    data = pd.DataFrame({
        'date': dates,
        'day_of_week': dates.dayofweek,
        'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
        'temperature': np.random.normal(20, 5, n_samples),
        'is_holiday': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'marketing_spend': np.random.uniform(1000, 5000, n_samples),
        'competitor_promo': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'special_event': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    })
    
    # Simulate demand based on features
    base_demand = 1000
    data['demand'] = (
        base_demand +
        data['is_weekend'] * 200 +
        data['temperature'] * 10 +
        data['is_holiday'] * 500 +
        data['marketing_spend'] * 0.1 +
        data['competitor_promo'] * -100 +
        data['special_event'] * 300 +
        np.random.normal(0, 50, n_samples)
    ).astype(int)
    
    return data