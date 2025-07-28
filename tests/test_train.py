"""
Unit tests for training pipeline
"""
import pytest
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.linear_model import LinearRegression
from utils import load_and_prepare_data
from train import train_model

class TestTrainingPipeline:
    
    def test_dataset_loading(self):
        """Test if dataset loads correctly"""
        X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
        
        # Check if data is loaded
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        assert scaler is not None
        
        # Check shapes
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        
        # Check if features are scaled
        assert abs(np.mean(X_train)) < 0.1  # Should be close to 0
        assert abs(np.std(X_train) - 1.0) < 0.1  # Should be close to 1
    
    def test_model_creation(self):
        """Test if model is LinearRegression instance"""
        X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
        model = LinearRegression()
        
        assert isinstance(model, LinearRegression)
    
    def test_model_training(self):
        """Test if model trains correctly"""
        X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Check if model has coefficients (indicating it was trained)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_ is not None
        assert model.intercept_ is not None
        
        # Check coefficient shape
        assert model.coef_.shape[0] == X_train.shape[1]
    
    def test_model_performance(self):
        """Test if model meets minimum performance threshold"""
        from sklearn.metrics import r2_score
        
        X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # Check if R² score exceeds minimum threshold (0.5)
        assert r2 > 0.5, f"R² score {r2:.4f} is below minimum threshold of 0.5"
        
        print(f"Model R² score: {r2:.4f}")
    
    def test_full_training_pipeline(self):
        """Test the complete training pipeline"""
        model, test_data, scaler = train_model()
        
        # Check if model is trained
        assert isinstance(model, LinearRegression)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        
        # Check if files are saved
        assert os.path.exists('model.joblib')
        assert os.path.exists('scaler.joblib')
        
        # Clean up
        if os.path.exists('model.joblib'):
            os.remove('model.joblib')
        if os.path.exists('scaler.joblib'):
            os.remove('scaler.joblib')
