from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data, predict_with_probability, get_species_name, get_model_info
import numpy as np

app = FastAPI(title="Enhanced Iris Classification API", version="2.0.0")

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    model_name: str = None  # Optional model selection

class IrisResponse(BaseModel):
    response: int
    species: str
    model_used: str

class IrisDetailedResponse(BaseModel):
    response: int
    species: str
    model_used: str
    probabilities: dict
    confidence: float

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    model_info = get_model_info()
    return {
        "status": "healthy",
        "message": "Enhanced Iris Classification API is running",
        "model": model_info.get('model_name', 'Unknown'),
        "model_type": model_info.get('model_type', 'Classifier'),
        "version": "2.0.0 - Multi-model enhanced",
        "classes": model_info.get('classes', ['setosa', 'versicolor', 'virginica'])
    }

@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    """
    Predict iris species using the specified model or best trained model.
    """
    try:
        features = [[
            iris_features.sepal_length, 
            iris_features.sepal_width,
            iris_features.petal_length, 
            iris_features.petal_width
        ]]

        prediction = predict_data(features, iris_features.model_name)
        species = get_species_name(int(prediction[0]))
        model_used = iris_features.model_name or get_model_info().get('best_model_name', 'Unknown')
        
        return IrisResponse(
            response=int(prediction[0]),
            species=species,
            model_used=model_used
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_detailed", response_model=IrisDetailedResponse)
async def predict_iris_detailed(iris_features: IrisData):
    """
    Predict iris species with detailed probability analysis using specified model.
    """
    try:
        features = [[
            iris_features.sepal_length,
            iris_features.sepal_width,
            iris_features.petal_length,
            iris_features.petal_width
        ]]

        prediction, probabilities = predict_with_probability(features, iris_features.model_name)
        species = get_species_name(int(prediction[0]))
        model_used = iris_features.model_name or get_model_info().get('best_model_name', 'Unknown')
        
        # Format probabilities
        prob_dict = {
            "Setosa": float(probabilities[0][0]),
            "Versicolor": float(probabilities[0][1]),
            "Virginica": float(probabilities[0][2])
        }
        
        # Calculate confidence (highest probability)
        confidence = float(max(probabilities[0]))
        
        return IrisDetailedResponse(
            response=int(prediction[0]),
            species=species,
            model_used=model_used,
            probabilities=prob_dict,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model_info")
async def get_api_model_info():
    """
    Get detailed information about the trained model.
    """
    model_info = get_model_info()
    return {
        "model_name": model_info.get('model_name', 'Unknown'),
        "model_type": model_info.get('model_type', 'Classifier'),
        "task": "Multi-class Classification",
        "features": model_info.get('features', ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']),
        "classes": model_info.get('classes', ['setosa', 'versicolor', 'virginica']),
        "feature_descriptions": {
            "sepal_length": "Length of the sepal in centimeters",
            "sepal_width": "Width of the sepal in centimeters", 
            "petal_length": "Length of the petal in centimeters",
            "petal_width": "Width of the petal in centimeters"
        },
        "class_descriptions": {
            "setosa": "Iris Setosa - typically smaller flowers with broader petals",
            "versicolor": "Iris Versicolor - medium-sized flowers with moderate proportions",
            "virginica": "Iris Virginica - typically larger flowers with longer petals"
        },
        "description": "Enhanced iris classification using multiple machine learning algorithms - automatically selects the best performing model"
    }

@app.get("/feature_ranges")
async def get_feature_ranges():
    """
    Get the typical ranges for iris features to help with input validation.
    """
    return {
        "sepal_length": {"min": 4.3, "max": 7.9, "unit": "cm"},
        "sepal_width": {"min": 2.0, "max": 4.4, "unit": "cm"},
        "petal_length": {"min": 1.0, "max": 6.9, "unit": "cm"},
        "petal_width": {"min": 0.1, "max": 2.5, "unit": "cm"}
    }