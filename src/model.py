from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class DemandPredictionModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.feature_columns = None
    
    def train(self, X, y):
        """
        Train the demand prediction model.
        
        Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target variable (demand)
        """
        self.feature_columns = X.columns
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
        X (pandas.DataFrame): Features
        
        Returns:
        numpy.array: Predicted demand
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model's performance.
        
        Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Actual demand
        
        Returns:
        dict: Performance metrics
        """
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model.
        
        Returns:
        pandas.DataFrame: Feature importance
        """
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)