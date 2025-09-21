import numpy as np
import joblib
import os
from train import run_training

# Global variables to store models
all_models = None
scaler = None
model_info = None

def load_models():
    """
    Load all models and metadata.
    """
    global all_models, scaler, model_info
    
    try:
        # Load all models
        all_models = joblib.load("model/all_models.pkl")
        print("All models loaded successfully")
        
        # Load scaler
        scaler = joblib.load("model/scaler.pkl")
        print("Scaler loaded successfully")
        
        # Load model info
        model_info = joblib.load("model/model_info.pkl")
        print("Model info loaded successfully")
        
        return True
    except Exception as e:
        print(f"Error loading enhanced models: {e}")
        return False

def load_fallback_model():
    """
    Load the original simple model as fallback.
    """
    try:
        model = joblib.load("model/model.pkl")
        print("Fallback model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading fallback model: {e}")
        return None

def predict_iris(sepal_length, sepal_width, petal_length, petal_width, model_name=None):
    """
    Predict iris species using specified model or best model.
    Args:
        sepal_length (float): Sepal length measurement
        sepal_width (float): Sepal width measurement
        petal_length (float): Petal length measurement
        petal_width (float): Petal width measurement
        model_name (str, optional): Name of specific model to use
    Returns:
        str: Predicted species name
    """
    global all_models, scaler, model_info
    
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Try to use enhanced models first
    if all_models is not None and scaler is not None:
        try:
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Select model
            if model_name and model_name in all_models:
                model = all_models[model_name]
            else:
                # Use best model
                best_model_name = model_info.get('best_model_name', 'Decision Tree')
                model = all_models[best_model_name]
            
            # Make prediction
            prediction = model.predict(input_scaled)
            
            # Convert numerical prediction to species name
            species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
            return species_map.get(prediction[0], f'species_{prediction[0]}')
            
        except Exception as e:
            print(f"Error with enhanced prediction: {e}")
    
    # Fallback to simple model
    fallback_model = load_fallback_model()
    if fallback_model is not None:
        try:
            prediction = fallback_model.predict(input_data)
            return prediction[0]
        except Exception as e:
            print(f"Error with fallback prediction: {e}")
    
    return "error"

def predict_iris_with_probability(sepal_length, sepal_width, petal_length, petal_width, model_name=None):
    """
    Predict iris species with probability scores.
    Returns:
        dict: Prediction results with probabilities
    """
    global all_models, scaler, model_info
    
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    if all_models is not None and scaler is not None:
        try:
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Select model
            if model_name and model_name in all_models:
                model = all_models[model_name]
                used_model = model_name
            else:
                # Use best model
                best_model_name = model_info.get('best_model_name', 'Decision Tree')
                model = all_models[best_model_name]
                used_model = best_model_name
            
            # Make prediction
            prediction = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)
            
            # Convert to readable format
            species_names = ['setosa', 'versicolor', 'virginica']
            predicted_species = species_names[prediction[0]]
            
            prob_dict = {
                species_names[i]: float(probabilities[0][i]) 
                for i in range(len(species_names))
            }
            
            return {
                'prediction': predicted_species,
                'probabilities': prob_dict,
                'confidence': float(max(probabilities[0])),
                'model_used': used_model
            }
            
        except Exception as e:
            print(f"Error with enhanced probability prediction: {e}")
    
    # Fallback to simple prediction
    simple_prediction = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    return {
        'prediction': simple_prediction,
        'probabilities': {},
        'confidence': 1.0,
        'model_used': 'Fallback Model'
    }

def get_model_info():
    """
    Get information about available models.
    """
    global model_info
    
    if model_info is not None:
        return model_info
    else:
        return {
            'best_model_name': 'Unknown',
            'available_models': ['Fallback Model'],
            'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            'classes': ['setosa', 'versicolor', 'virginica']
        }

def get_available_models():
    """
    Get list of available model names.
    """
    global all_models
    
    if all_models is not None:
        return list(all_models.keys())
    else:
        return ['Fallback Model']

# Initialize models on import
if __name__ == "__main__":
    # Check if enhanced models exist
    if os.path.exists("model/all_models.pkl"):
        if load_models():
            print("Enhanced models ready")
        else:
            print("Using fallback mode")
    else:
        # Check if we have basic model or need to train
        if os.path.exists("model/model.pkl"):
            print("Basic model found")
        else:
            print("No models found. Training new models...")
            os.makedirs("model", exist_ok=True)
            run_training()
            load_models()
else:
    # When imported, try to load enhanced models
    if os.path.exists("model/all_models.pkl"):
        load_models()