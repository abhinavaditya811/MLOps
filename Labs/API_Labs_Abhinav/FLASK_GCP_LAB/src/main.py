from flask import Flask, request, jsonify
from predict import predict_iris, predict_iris_with_probability, get_model_info, get_available_models
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    model_info = get_model_info()
    return jsonify({
        'status': 'healthy',
        'message': 'Enhanced Iris Classification Flask API is running',
        'version': '2.0.0',
        'best_model': model_info.get('best_model_name', 'Unknown'),
        'available_models': model_info.get('available_models', []),
        'endpoints': ['/predict', '/predict_detailed', '/models', '/model_info']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Basic prediction endpoint.
    """
    try:
        data = request.get_json()
        
        # Extract features
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
        
        # Optional model selection
        model_name = data.get('model_name', None)
        
        print(f"Predicting with: {sepal_length}, {sepal_width}, {petal_length}, {petal_width}")
        if model_name:
            print(f"Using model: {model_name}")
        
        # Make prediction
        prediction = predict_iris(sepal_length, sepal_width, petal_length, petal_width, model_name)
        
        return jsonify({
            'prediction': prediction,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/predict_detailed', methods=['POST'])
def predict_detailed():
    """
    Detailed prediction endpoint with probabilities.
    """
    try:
        data = request.get_json()
        
        # Extract features
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
        
        # Optional model selection
        model_name = data.get('model_name', None)
        
        print(f"Detailed prediction with: {sepal_length}, {sepal_width}, {petal_length}, {petal_width}")
        if model_name:
            print(f"Using model: {model_name}")
        
        # Make detailed prediction
        result = predict_iris_with_probability(sepal_length, sepal_width, petal_length, petal_width, model_name)
        
        return jsonify({
            'prediction': result['prediction'],
            'probabilities': result['probabilities'],
            'confidence': result['confidence'],
            'model_used': result['model_used'],
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/models', methods=['GET'])
def get_models():
    """
    Get available models endpoint.
    """
    try:
        available_models = get_available_models()
        model_info = get_model_info()
        
        return jsonify({
            'available_models': available_models,
            'best_model': model_info.get('best_model_name', 'Unknown'),
            'total_models': len(available_models),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Get detailed model information endpoint.
    """
    try:
        info = get_model_info()
        
        return jsonify({
            'best_model_name': info.get('best_model_name', 'Unknown'),
            'best_model_type': info.get('best_model_type', 'Unknown'),
            'available_models': info.get('available_models', []),
            'features': info.get('features', []),
            'classes': info.get('classes', []),
            'feature_descriptions': {
                'sepal_length': 'Length of the sepal in centimeters',
                'sepal_width': 'Width of the sepal in centimeters',
                'petal_length': 'Length of the petal in centimeters',
                'petal_width': 'Width of the petal in centimeters'
            },
            'class_descriptions': {
                'setosa': 'Iris Setosa - typically smaller flowers',
                'versicolor': 'Iris Versicolor - medium-sized flowers',
                'virginica': 'Iris Virginica - typically larger flowers'
            },
            'description': 'Enhanced iris classification using multiple ML algorithms',
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/predict', '/predict_detailed', '/models', '/model_info'],
        'status': 'error'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed',
        'message': 'Check the HTTP method (GET/POST) for this endpoint',
        'status': 'error'
    }), 405

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))