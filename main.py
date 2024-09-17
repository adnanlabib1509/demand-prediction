import pandas as pd
import os
from src.data_generator import generate_synthetic_data
from src.feature_engineering import engineer_features
from src.model import DemandPredictionModel
from src.visualization import plot_actual_vs_predicted, plot_feature_importance, plot_demand_over_time
from sklearn.model_selection import train_test_split

def main():
    # Generate synthetic data
    data = generate_synthetic_data()
    
    # Save the generated data to a CSV file
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_file_path = os.path.join(data_dir, 'synthetic_data.csv')
    data.to_csv(data_file_path, index=False)
    print(f"Synthetic data saved to {data_file_path}")
    
    # Perform feature engineering
    engineered_data = engineer_features(data)
    
    # Prepare features and target
    features = ['day_of_week', 'is_weekend', 'temperature', 'is_holiday', 'marketing_spend',
                'competitor_promo', 'special_event', 'demand_lag1', 'demand_lag7',
                'demand_rolling_mean_7', 'demand_rolling_mean_30', 'temp_weekend_interaction',
                'marketing_holiday_interaction']
    target = 'demand'
    
    X = engineered_data[features]
    y = engineered_data[target]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = DemandPredictionModel()
    model.train(X_train, y_train)
    
    # Evaluate the model
    performance = model.evaluate(X_test, y_test)
    print("Model Performance:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Visualize results
    plot_actual_vs_predicted(y_test, y_pred)
    plot_feature_importance(model.get_feature_importance())
    plot_demand_over_time(data)

if __name__ == "__main__":
    main()