"""
Training script for Linear Regression model
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from utils import load_and_prepare_data, save_model_artifacts

def train_model():
    """
    Train Linear Regression model on California Housing dataset
    
    Returns:
        tuple: trained model, test data, scaler
    """
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    print(f"Training R² score: {train_r2:.4f}")
    print(f"Test R² score: {test_r2:.4f}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    
    # Save model and scaler
    save_model_artifacts(model, scaler)
    
    return model, (X_test, y_test), scaler

if __name__ == "__main__":
    model, test_data, scaler = train_model()
    print("Training completed successfully!")
