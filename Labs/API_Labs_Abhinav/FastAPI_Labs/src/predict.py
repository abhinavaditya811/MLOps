import joblib
import numpy as np

def predict_data(X, model_name=None):
    """
    Predict the iris species for the input data using specified model.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
        model_name (str, optional): Name of specific model to use. If None, uses best model.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    if model_name:
        # Load all models and use specific one
        all_models = joblib.load("../model/all_models.pkl")
        if model_name not in all_models:
            raise ValueError(f"Model {model_name} not found. Available: {list(all_models.keys())}")
        model = all_models[model_name]
    else:
        # Load the best model (default behavior)
        model = joblib.load("../model/iris_model.pkl")
    
    # Load the scaler
    scaler = joblib.load("../model/iris_scaler.pkl")
    
    # Scale the input data
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    return y_pred

def predict_with_probability(X, model_name=None):
    """
    Predict iris species with probability scores using specified model.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
        model_name (str, optional): Name of specific model to use. If None, uses best model.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
        y_proba (numpy.ndarray): Prediction probabilities.
    """
    if model_name:
        # Load all models and use specific one
        all_models = joblib.load("../model/all_models.pkl")
        if model_name not in all_models:
            raise ValueError(f"Model {model_name} not found. Available: {list(all_models.keys())}")
        model = all_models[model_name]
    else:
        # Load the best model (default behavior)
        model = joblib.load("../model/iris_model.pkl")
    
    # Load the scaler
    scaler = joblib.load("../model/iris_scaler.pkl")
    
    # Scale the input data
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    
    return y_pred, y_proba

def get_species_name(prediction):
    """
    Convert numerical prediction to species name.
    Args:
        prediction (int): Predicted class (0, 1, or 2).
    Returns:
        str: Species name.
    """
    species_names = {
        0: "Iris Setosa",
        1: "Iris Versicolor", 
        2: "Iris Virginica"
    }
    return species_names.get(prediction, "Unknown")

def get_model_info():
    """
    Get information about the trained model.
    Returns:
        dict: Model information.
    """
    try:
        model_info = joblib.load("../model/model_info.pkl")
        return model_info
    except:
        return {
            'model_name': 'Unknown',
            'model_type': 'Classifier',
            'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            'classes': ['setosa', 'versicolor', 'virginica']
        }