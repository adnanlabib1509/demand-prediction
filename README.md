# Deliveroo Demand Prediction Project

## Overview

This project demonstrates a comprehensive approach to predicting food delivery demand for a service like Deliveroo. It showcases various data science and machine learning techniques, including data generation, feature engineering, model training, evaluation, and visualization.

## Project Structure

```
demand_prediction/
├── data/
│   └── synthetic_data.csv
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── visualization.py
├── main.py
├── requirements.txt
└── README.md
```

## Features

1. **Synthetic Data Generation**: Creates realistic food delivery demand data, considering factors like day of the week, temperature, holidays, marketing spend, competitor promotions, and special events. The generated data is saved to a CSV file for consistency and reproducibility.

2. **Feature Engineering**: Implements advanced feature engineering techniques, including:
   - Lag features
   - Rolling mean features
   - Interaction features
   - Feature scaling

3. **Machine Learning Model**: Utilizes a Random Forest Regressor for demand prediction, chosen for its ability to capture complex relationships and provide feature importance.

4. **Model Evaluation**: Assesses the model's performance using multiple metrics:
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R-squared (R2) score

5. **Visualization**: Includes several visualization functions to help interpret the results:
   - Actual vs. Predicted demand plot
   - Feature importance plot
   - Demand over time plot

6. **Modular Design**: The project is structured in a modular way, making it easy to understand, maintain, and extend.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/deliveroo-demand-prediction.git
   cd deliveroo-demand-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:
   ```
   python main.py
   ```

   This will generate synthetic data and save it to `data/synthetic_data.csv`, train the model, evaluate its performance, and display visualizations.

## Key Components

### Data Generator (`src/data_generator.py`)

Generates synthetic data mimicking real-world food delivery demand patterns. It considers various factors that might influence demand, providing a rich dataset for analysis and model training. The generated data is saved to a CSV file for reproducibility.

### Feature Engineering (`src/feature_engineering.py`)

Implements advanced feature engineering techniques to extract more information from the raw data:
- Creates lag features to capture time-dependent patterns
- Calculates rolling mean features to smooth out short-term fluctuations
- Generates interaction features to capture combined effects of different variables
- Normalizes numerical features to ensure all features are on the same scale

### Model (`src/model.py`)

Defines a `DemandPredictionModel` class that encapsulates the machine learning model:
- Uses Random Forest Regressor as the underlying algorithm
- Provides methods for training, prediction, and evaluation
- Includes a method to extract feature importance

### Visualization (`src/visualization.py`)

Contains functions to create insightful visualizations:
- Actual vs. Predicted demand scatter plot
- Feature importance bar plot
- Demand over time line plot

## Future Improvements

1. Implement more advanced time series techniques (e.g., ARIMA, Prophet)
2. Incorporate external data sources (e.g., weather forecasts, local events calendars)
3. Develop a real-time prediction system
4. Create a web interface for easy interaction with the model
5. Implement automated model retraining and monitoring
6. Add option to load existing data instead of generating new data each time

## License

This project is licensed under the MIT License - see the LICENSE file for details.
