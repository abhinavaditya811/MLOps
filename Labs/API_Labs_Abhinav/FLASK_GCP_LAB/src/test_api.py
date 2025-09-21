import requests
import json

# Flask backend URL
BASE_URL = 'http://127.0.0.1:8080'

def test_health_check():
    """Test the health check endpoint."""
    print("=" * 50)
    print("Testing Health Check Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f'{BASE_URL}/')
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Service: {data.get('message', 'Unknown')}")
            print(f"Version: {data.get('version', 'Unknown')}")
            print(f"Best Model: {data.get('best_model', 'Unknown')}")
            print(f"Available Models: {data.get('available_models', [])}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {e}")

def test_basic_prediction():
    """Test basic prediction endpoint."""
    print("\n" + "=" * 50)
    print("Testing Basic Prediction Endpoint")
    print("=" * 50)
    
    # Test data - typical Iris setosa
    data = {
        'sepal_length': 5.1,
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2
    }
    
    try:
        response = requests.post(f'{BASE_URL}/predict', json=data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted Species: {result.get('prediction', 'Unknown')}")
            print(f"Status: {result.get('status', 'Unknown')}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {e}")

def test_detailed_prediction():
    """Test detailed prediction endpoint."""
    print("\n" + "=" * 50)
    print("Testing Detailed Prediction Endpoint")
    print("=" * 50)
    
    # Test data - typical Iris virginica
    data = {
        'sepal_length': 6.3,
        'sepal_width': 3.3,
        'petal_length': 6.0,
        'petal_width': 2.5
    }
    
    try:
        response = requests.post(f'{BASE_URL}/predict_detailed', json=data)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted Species: {result.get('prediction', 'Unknown')}")
            print(f"Confidence: {result.get('confidence', 0):.1%}")
            print(f"Model Used: {result.get('model_used', 'Unknown')}")
            
            probabilities = result.get('probabilities', {})
            if probabilities:
                print("Probabilities:")
                for species, prob in probabilities.items():
                    print(f"  {species}: {prob:.3f}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {e}")

def test_model_selection():
    """Test prediction with specific model selection."""
    print("\n" + "=" * 50)
    print("Testing Model Selection")
    print("=" * 50)
    
    # First get available models
    try:
        models_response = requests.get(f'{BASE_URL}/models')
        if models_response.status_code == 200:
            models_info = models_response.json()
            available_models = models_info.get('available_models', [])
            print(f"Available Models: {available_models}")
            
            if available_models:
                # Test with specific model
                test_model = available_models[0]  # Use first available model
                print(f"\nTesting with model: {test_model}")
                
                data = {
                    'sepal_length': 5.8,
                    'sepal_width': 2.7,
                    'petal_length': 4.1,
                    'petal_width': 1.0,
                    'model_name': test_model
                }
                
                response = requests.post(f'{BASE_URL}/predict_detailed', json=data)
                if response.status_code == 200:
                    result = response.json()
                    print(f"Prediction: {result.get('prediction', 'Unknown')}")
                    print(f"Model Used: {result.get('model_used', 'Unknown')}")
                    print(f"Confidence: {result.get('confidence', 0):.1%}")
                else:
                    print(f"Prediction Error: {response.text}")
        else:
            print(f"Models Error: {models_response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {e}")

def test_model_info():
    """Test model info endpoint."""
    print("\n" + "=" * 50)
    print("Testing Model Info Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f'{BASE_URL}/model_info')
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            info = response.json()
            print(f"Best Model: {info.get('best_model_name', 'Unknown')}")
            print(f"Model Type: {info.get('best_model_type', 'Unknown')}")
            print(f"Features: {info.get('features', [])}")
            print(f"Classes: {info.get('classes', [])}")
            print(f"Description: {info.get('description', 'No description')}")
        else:
            print(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {e}")

def test_different_species():
    """Test predictions for different iris species."""
    print("\n" + "=" * 50)
    print("Testing Different Iris Species")
    print("=" * 50)
    
    test_cases = [
        {
            'name': 'Typical Setosa',
            'data': {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2}
        },
        {
            'name': 'Typical Versicolor', 
            'data': {'sepal_length': 5.9, 'sepal_width': 3.0, 'petal_length': 4.2, 'petal_width': 1.5}
        },
        {
            'name': 'Typical Virginica',
            'data': {'sepal_length': 6.3, 'sepal_width': 3.3, 'petal_length': 6.0, 'petal_width': 2.5}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        try:
            response = requests.post(f'{BASE_URL}/predict_detailed', json=test_case['data'])
            if response.status_code == 200:
                result = response.json()
                print(f"  Prediction: {result.get('prediction', 'Unknown')}")
                print(f"  Confidence: {result.get('confidence', 0):.1%}")
            else:
                print(f"  Error: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"  Connection Error: {e}")

if __name__ == '__main__':
    print("Enhanced Flask Iris Classification API Test Suite")
    print("Make sure the Flask server is running on http://127.0.0.1:8080")
    
    # Run all tests
    test_health_check()
    test_basic_prediction()
    test_detailed_prediction()
    test_model_selection()
    test_model_info()
    test_different_species()
    
    print("\n" + "=" * 50)
    print("Test Suite Complete!")
    print("=" * 50)