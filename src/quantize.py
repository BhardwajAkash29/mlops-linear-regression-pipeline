"""
Manual quantization script for Linear Regression model
"""
import numpy as np
import joblib
from utils import load_model_artifacts

def quantize_parameters(params, bits=8):
    """
    Quantize parameters to specified bits
    
    Args:
        params (np.array): Parameters to quantize
        bits (int): Number of bits for quantization
        
    Returns:
        tuple: quantized_params, scale, zero_point
    """
    # Calculate scale and zero point for quantization
    param_min = np.min(params)
    param_max = np.max(params)
    
    # Calculate scale
    scale = (param_max - param_min) / (2**bits - 1)
    
    # Calculate zero point
    zero_point = -param_min / scale
    zero_point = np.round(zero_point).astype(np.uint8)
    
    # Quantize parameters
    quantized = np.round(params / scale + zero_point).astype(np.uint8)
    
    return quantized, scale, zero_point

def dequantize_parameters(quantized_params, scale, zero_point):
    """
    Dequantize parameters back to float32
    
    Args:
        quantized_params (np.array): Quantized parameters
        scale (float): Scale factor
        zero_point (int): Zero point
        
    Returns:
        np.array: Dequantized parameters
    """
    dequantized = (quantized_params.astype(np.float32) - zero_point) * scale
    return dequantized

def quantize_model():
    """
    Load trained model and perform manual quantization
    """
    print("Loading trained model...")
    try:
        model, scaler = load_model_artifacts()
    except FileNotFoundError:
        print("Error: Model not found. Please run training first.")
        return
    
    # Extract parameters
    coef = model.coef_
    intercept = model.intercept_
    
    print(f"Original coefficients shape: {coef.shape}")
    print(f"Original intercept: {intercept}")
    
    # Save original parameters
    original_params = {
        'coef': coef,
        'intercept': intercept
    }
    joblib.dump(original_params, 'unquant_params.joblib')
    print("Original parameters saved to unquant_params.joblib")
    
    # Quantize coefficients
    print("Quantizing coefficients...")
    quant_coef, coef_scale, coef_zero_point = quantize_parameters(coef)
    
    # Quantize intercept
    print("Quantizing intercept...")
    quant_intercept, intercept_scale, intercept_zero_point = quantize_parameters(
        np.array([intercept])
    )
    
    # Save quantized parameters
    quantized_params = {
        'quant_coef': quant_coef,
        'coef_scale': coef_scale,
        'coef_zero_point': coef_zero_point,
        'quant_intercept': quant_intercept[0],
        'intercept_scale': intercept_scale,
        'intercept_zero_point': intercept_zero_point
    }
    joblib.dump(quantized_params, 'quant_params.joblib')
    print("Quantized parameters saved to quant_params.joblib")
    
    # Dequantize and test
    print("Testing dequantization...")
    dequant_coef = dequantize_parameters(quant_coef, coef_scale, coef_zero_point)
    dequant_intercept = dequantize_parameters(
        quant_intercept, intercept_scale, intercept_zero_point
    )[0]
    
    # Calculate quantization error
    coef_error = np.mean(np.abs(coef - dequant_coef))
    intercept_error = abs(intercept - dequant_intercept)
    
    print(f"Coefficient quantization error: {coef_error:.6f}")
    print(f"Intercept quantization error: {intercept_error:.6f}")
    
    # Test inference with dequantized weights
    print("Testing inference with dequantized weights...")
    from utils import load_and_prepare_data
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data()
    
    # Original prediction
    original_pred = model.predict(X_test[:5])
    
    # Dequantized prediction
    dequant_pred = np.dot(X_test[:5], dequant_coef) + dequant_intercept
    
    print("Sample predictions comparison:")
    print("Original:", original_pred)
    print("Dequantized:", dequant_pred)
    print("Max difference:", np.max(np.abs(original_pred - dequant_pred)))

if __name__ == "__main__":
    quantize_model()
    print("Quantization completed successfully!")
