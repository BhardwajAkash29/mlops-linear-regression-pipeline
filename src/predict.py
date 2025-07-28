"""
Prediction script for model verification
"""
import numpy as np
from utils import load_model_artifacts, load_and_prepare_data

def make_predictions():
    """
    Load model and make predictions on test set
    """
    try:
        print("Loading trained model...")
        model, scaler = load_model_artifacts()
        
        print("Loading test data...")
        X_train, X_test, y_train, y_test, _ = load_and_prepare_data()
        
        print("Making predictions...")
        predictions = model.predict(X_test)
        
        print("Sample predictions vs actual values:")
        for i in range(min(10, len(predictions))):
            print(f"Prediction: {predictions[i]:.4f}, Actual: {y_test[i]:.4f}")
        
        # Calculate basic statistics
        print(f"\nPrediction statistics:")
        print(f"Mean prediction: {np.mean(predictions):.4f}")
        print(f"Std prediction: {np.std(predictions):.4f}")
        print(f"Min prediction: {np.min(predictions):.4f}")
        print(f"Max prediction: {np.max(predictions):.4f}")
        
        return predictions
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

if __name__ == "__main__":
    predictions = make_predictions()
    if predictions is not None:
        print("Prediction completed successfully!")
    else:
        print("Prediction failed!")
