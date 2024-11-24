from flask import Flask, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import logging
import os
from keras import backend as K

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the model
model = None

def recall_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def f1_m(y_true, y_pred):
  precision = precision_m(y_true, y_pred)
  recall = recall_m(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

@app.route('/api/model/load', methods=['GET'])
def load_model():
    global model
    try:
        model_path = os.path.join('api', 'model', 'vivit_model')
        if not os.path.exists(model_path):
            logger.error("Model file not found")
            return jsonify({
                'status': 'error',
                'message': 'Model file not found',
                'model_loaded': False
            }), 404

        model = keras.models.load_model(
            model_path,
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

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to load model: {str(e)}',
            'model_loaded': False
        }), 500

@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    global model
    return jsonify({
        'model_loaded': model is not None
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
        'models_loaded': model is not None
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)