from flask import Flask, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import logging
import os

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = "model/vivit_model"

# Global variable to store the model
model = None

def recall_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))

@app.route('/api/model/load', methods=['GET'])
def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model not found at path: {MODEL_PATH}")
            return jsonify({
                'status': 'error',
                'message': 'Model file not found',
                'model_loaded': False
            }), 404

        try:
            model = keras.models.load_model(
                MODEL_PATH,
                custom_objects={
                    'recall_m': recall_m,
                    'precision_m': precision_m,
                    'f1_m': f1_m
                }
            )
            logger.info("Model loaded successfully")
            return jsonify({
                'status': 'success',
                'message': 'Model loaded successfully',
                'model_loaded': True
            })
        except Exception as model_error:
            logger.error(f"Error loading model: {str(model_error)}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to load model: {str(model_error)}',
                'model_loaded': False
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Unexpected error: {str(e)}',
            'model_loaded': False
        }), 500

@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    global model
    return jsonify({
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

@app.route('/api/features', methods=['GET'])
def get_features():
    features = {
        'webcam': {
            'name': 'Live Webcam',
            'description': 'Real-time monitoring using your device\'s camera',
            'status': 'active',
            'capabilities': ['face-detection', 'motion-tracking', 'incident-reporting']
        },
        'upload': {
            'name': 'Video Upload',
            'description': 'Analyze pre-recorded videos for incidents',
            'status': 'active',
            'capabilities': ['batch-processing', 'frame-analysis', 'export-results']
        },
        'youtube': {
            'name': 'YouTube Stream',
            'description': 'Monitor YouTube streams and videos',
            'status': 'active',
            'capabilities': ['stream-analysis', 'real-time-alerts', 'url-processing']
        }
    }
    return jsonify(features)

@app.route('/api/status', methods=['GET'])
def get_status():
    global model
    return jsonify({
        'status': 'operational',
        'version': '1.0.0',
        'models_loaded': model is not None,
        'model_path': MODEL_PATH
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)