from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import logging
import os
import cv2
import numpy as np
import time
import base64

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = "model/vivit_model"

# Global variables
model = None
frames_buffer = []
SEGMENT_FRAMES = 42

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

def process_video_feed(frame, model, frames_buffer, segment_frames=42):
    """Process video frame and make prediction with enhanced debugging"""
    try:
        resized_frame = cv2.resize(frame, (128, 128))
        frames_buffer.append(resized_frame)
        
        if len(frames_buffer) > segment_frames:
            frames_buffer.pop(0)
        
        if len(frames_buffer) == segment_frames:
            video_segment = np.array(frames_buffer)
            video_segment = video_segment.astype('float32') / 255.0
            video_segment = np.expand_dims(video_segment, axis=0)
            
            prediction = model.predict(video_segment, verbose=0)
            
            # Get probabilities for both classes
            no_fight_prob = prediction[0][0]
            fight_prob = prediction[0][1]
            
            # Use fight probability for confidence
            predicted_class = 1 if fight_prob > 0.4 else 0
            confidence = fight_prob
            
            logger.debug(f"\nPrediction Analysis:")
            logger.debug(f"No Fight Probability: {no_fight_prob:.2%}")
            logger.debug(f"Fight Probability: {fight_prob:.2%}")
            logger.debug(f"Predicted class: {'Fight' if predicted_class == 1 else 'No Fight'}")
            logger.debug(f"Confidence: {confidence:.2%}")
            
            return predicted_class, confidence
        else:
            logger.debug(f"\rCollecting frames: {len(frames_buffer)}/{segment_frames}")
        
        return None, 0.0
        
    except Exception as e:
        logger.error(f"Error in process_video_feed: {str(e)}")
        return None, 0.0

@app.route('/api/webcam/predict', methods=['POST'])
def predict_webcam():
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 400

    try:
        # Get base64 frame from request
        frame_data = request.json.get('frame')
        if not frame_data:
            return jsonify({
                'error': 'No frame data provided'
            }), 400

        # Decode base64 frame
        frame_bytes = base64.b64decode(frame_data)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process frame
        predicted_class, confidence = process_video_feed(
            frame, model, frames_buffer, SEGMENT_FRAMES
        )
        print("hello : ",predicted_class)
        if predicted_class is not None:
            return jsonify({
                'predicted_class': int(predicted_class),
                'confidence': float(confidence)
            })
        else:
            return jsonify({
                'status': 'collecting_frames',
                'frames_collected': len(frames_buffer),
                'frames_required': SEGMENT_FRAMES
            })

    except Exception as e:
        logger.error(f"Error processing webcam frame: {str(e)}")
        return jsonify({
            'error': f'Error processing frame: {str(e)}'
        }), 500
    
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

import base64


def monitor_video_file(video_data):
    """Process uploaded video file and make predictions"""
    try:
        # Decode base64 video data
        video_bytes = base64.b64decode(video_data)
        
        # Save temporary file
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
        
        # Open video file
        cap = cv2.VideoCapture(temp_path)
        frames = []
        predictions = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame
            resized_frame = cv2.resize(frame, (128, 128))
            frames.append(resized_frame)
            
            # Process when we have enough frames
            if len(frames) == SEGMENT_FRAMES:
                video_segment = np.array(frames)
                video_segment = video_segment.astype('float32') / 255.0
                video_segment = np.expand_dims(video_segment, axis=0)
                
                prediction = model.predict(video_segment, verbose=0)
                predictions.append({
                    'timestamp': len(predictions) * (SEGMENT_FRAMES / 30),  # Approximate timestamp
                    'fight_probability': float(prediction[0][1]),
                    'no_fight_probability': float(prediction[0][0])
                })
                
                frames = frames[SEGMENT_FRAMES//2:]  # Overlap segments
        
        cap.release()
        os.remove(temp_path)  # Clean up
        
        return {
            'status': 'success',
            'predictions': predictions
        }
        
    except Exception as e:
        logger.error(f"Error in monitor_video_file: {str(e)}")
        return {
            'status': 'error',
            'message': str(e)
        }

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 400
        
    try:
        video_data = request.json.get('video')
        if not video_data:
            return jsonify({
                'status': 'error',
                'message': 'No video data provided'
            }), 400
            
        results = monitor_video_file(video_data)
        return jsonify(results)
            
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error analyzing video: {str(e)}'
        }), 500

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