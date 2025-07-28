"""
Utility functions for the MLOps pipeline
"""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_prepare_data():
    """
    Load and prepare the California Housing dataset
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    # Load dataset
    california_housing = fetch_california_housing()
    X, y = california_housing.data, california_housing.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def ensure_directory_exists(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory (str): Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model_artifacts(model, scaler, model_path="model.joblib", scaler_path="scaler.joblib"):
    """
    Save model and scaler artifacts
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        model_path (str): Path to save model
        scaler_path (str): Path to save scaler
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

def load_model_artifacts(model_path="model.joblib", scaler_path="scaler.joblib"):
    """
    Load model and scaler artifacts
    
    Args:
        model_path (str): Path to load model from
        scaler_path (str): Path to load scaler from
        
    Returns:
        tuple: model, scaler
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
