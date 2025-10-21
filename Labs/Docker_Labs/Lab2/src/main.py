from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__, static_folder='statics')
CORS(app)  # Enable CORS for all routes

# Load the TensorFlow model
model = tf.keras.models.load_model('my_model.keras')
class_labels = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/')
def home():
    return "Welcome to the Iris Classifier API!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Handle both JSON and form data
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form.to_dict()

            sepal_length = float(data['sepal_length'])
            sepal_width = float(data['sepal_width'])
            petal_length = float(data['petal_length'])
            petal_width = float(data['petal_width'])

            # Perform the prediction
            input_data = np.array([sepal_length, sepal_width, petal_length, petal_width])[np.newaxis, ]
            prediction = model.predict(input_data)
            predicted_class = class_labels[np.argmax(prediction)]

            # Return the predicted class in the response
            return jsonify({"predicted_class": predicted_class})
        except Exception as e:
            return jsonify({"error": str(e)})
    elif request.method == 'GET':
        return render_template('predict.html')
    else:
        return "Unsupported HTTP method"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)