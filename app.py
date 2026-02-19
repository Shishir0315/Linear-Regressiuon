from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'regression_model.h5')
model = None

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        else:
            print(f"Model not found at {MODEL_PATH}")
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        x_val = float(data.get('x', 0))
        
        m = get_model()
        if m is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        prediction = m.predict(np.array([[x_val]]))
        result = float(prediction[0][0])
        
        return jsonify({
            'x': x_val,
            'prediction': round(result, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Initial load attempt
    get_model()
    app.run(host='0.0.0.0', port=7860)
