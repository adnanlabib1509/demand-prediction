import matplotlib.pyplot as plt
import seaborn as sns

def plot_actual_vs_predicted(y_true, y_pred):
    """
    Plot actual vs predicted values.
    
    Args:
    y_true (numpy.array): Actual values
    y_pred (numpy.array): Predicted values
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.title("Actual vs Predicted Demand")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_importance):
    """
    Plot feature importance.
    
    Args:
    feature_importance (pandas.DataFrame): Feature importance data
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

def plot_demand_over_time(data):
    """
    Plot demand over time.
    
    Args:
    data (pandas.DataFrame): Data containing 'date' and 'demand' columns
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['demand'])
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.title("Demand Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()