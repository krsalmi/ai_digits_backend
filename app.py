"""
This Flask application provides a web service for a digit recognition system 
using a convolutional neural network (CNN). 
The service allows users to train the model, predict digits from images, 
and monitor the training progress and model status.
The application uses Flask for the web framework, Redis for progress tracking
and status updates, and several other libraries such as NumPy and PIL for 
image processing and model handling.

Endpoints:
- `/`: Health check endpoint returning the status of the service.
- `/api/drawing/`: Endpoint to predict the digit from a given drawing.
- `/api/retrain_model/`: Initiates the retraining of the model.
- `/api/stop_training/`: Stops the ongoing model training process.
- `/api/check_model_status/`: Checks and returns the current status of the model.
- `/api/training_progress/`: Provides the current progress of the ongoing model training.
- `/api/model_accuracy/`: Returns the accuracy of the trained model.
"""

from flask import Flask, request, jsonify, stream_with_context, Response
from flask_cors import CORS
import os
import threading
import redis
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from digit_trainer import DigitModelTrainer
from config import Dev, Prod
import time



train_model_lock = threading.Lock() #global variable to lock the thread that is training the model, so multiple similar processes won't run at the same time
stop_training_event = threading.Event() # Event that will signal the training process to stop


app = Flask(__name__)

# Retrieve environment variable
env = os.getenv('FLASK_ENV', 'development')

if env == 'development':
    app.config.from_object(Dev)
else:
    app.config.from_object(Prod)

# Set cors policy
CORS(app, origins=[app.config['FRONTEND_URL']])

#Connect to an instance of Redis
if app.config['REDIS_URL']: # For prod environment
    redis_conn = redis.StrictRedis.from_url(app.config['REDIS_URL'])
else:
    redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)
redis_conn.set('training_progress', '{}')
redis_conn.set('model_status', app.config['MODEL_CREATION_STATUS']['NOT_STARTED'])
redis_conn.set('model_accuracy', 0)

# Initialize the model trainer
trainer = DigitModelTrainer(env=env)
# Load model
trainer.load_model()


@app.route("/")
def home():
    return jsonify({"status": "healty"})

@app.route('/api/drawing/', methods=['POST'])
def send_drawing():
    """
    Sends user drawing to model for digit prediction.
    Decodes the base64-encoded drawing data, preprocesses it into the 
    expected format for the model, runs prediction, and returns the 
    prediction result and confidence score.

    Returns:
        JSON response with prediction and confidence score if successful.
        Error responses if drawing can't be processed or model not available.
    """
    if not os.path.isfile(os.path.join(app.config['CUR_FOLDER'], app.config['MODEL_FILE'])):
        return jsonify({'message': 'Model file does not exist. Please create the model first.'}), 404
    try:
        drawing = request.json['drawing']
        decoded_drawing = decode_drawing(drawing)
        preprocessed_drawing = preprocess_drawing(decoded_drawing)
        prediction, confidence = trainer.predict_digit(preprocessed_drawing)
        if prediction is None or confidence is None:
            return jsonify({'error': "Model does not exist"}), 503
        return jsonify({'prediction': prediction, 'confidence': float(confidence)})
    except Exception as e:
        print(str(e))
        return jsonify({'error': 'An error occurred'}), 500
    


def decode_drawing(image_data):
    """
    Decodes the base64-encoded drawing data from the client.

    Removes the 'data:image/png;base64,' prefix, base64-decodes the data, 
    and converts it to a PIL Image object that can be used for prediction.

    Args:
        image_data (str): The base64-encoded PNG drawing data.

    Returns:
        Image: The decoded PNG image as a PIL Image object.
    """
    base64_data = image_data.split(',')[1]
    byte_data = base64.b64decode(base64_data)
    drawing = Image.open(BytesIO(byte_data))
    return drawing

def preprocess_drawing(drawing):
    """
    Preprocesses the decoded PNG drawing into the expected format for the model.

    Resizes, converts to grayscale, normalizes the pixel values, and 
    converts the image to a numpy array.

    Args:
        drawing (Image): The decoded PNG drawing 

    Returns:
        numpy.ndarray: The preprocessed grayscale drawing as a 28x28 numpy array
                        with values between 0-1.
    """
    drawing = drawing.resize((28, 28)).convert('L') 
    image_arr = np.array(drawing) / 255.0  # Convert to numpy array and normalize
    return image_arr
    
@app.route('/api/retrain_model/', methods=['POST'])
def retrain_model():
    """
    Trains a new digit recognition model.

    Acquires a lock to ensure only one model training can happen at a time.
    Starts model training in a separate thread and returns an accepted response 
    immediately to allow the client to poll for status updates.

    Returns:
        202 if model training successfully started.
        409 if model training is already in progress, this will avoid a race condition
    """
    global train_model_lock, stop_training_event

    if not train_model_lock.acquire(blocking=False):
        return jsonify({'message': 'Training is already in progress. Please try again later.'}), 409  # 409 Conflict
    
    stop_training_event.clear()
    # Start the model creation in a separate thread, so response can be sent immediately
    thread = threading.Thread(target=build_model)
    thread.start()
    redis_conn.set('model_status', app.config['MODEL_CREATION_STATUS']['IN_PROGRESS'])
    return jsonify({'message': 'Model creation started'}), 202

@app.route('/api/stop_training/', methods=['POST'])
def stop_training():
    """
    Stops an in-progress model training by setting a stop signal.

    Checks if a model training is currently in progress by checking
    the status in Redis. If training is in progress, it sets the 
    stop_training_event to signal the training thread to stop.

    Returns:
        200 if stop signal was successfully set.
        409 if training not in progress.
    """
    global stop_training_event
    # Check if training is in progress
    current_status = redis_conn.get('model_status').decode("utf-8")
    if current_status != app.config['MODEL_CREATION_STATUS']['IN_PROGRESS']: # If not "in_progress"
        return jsonify({'message': 'Training not in progress.'})
    
    stop_training_event.set()
    return jsonify({'message': 'Stop signal sent.'})

    
@app.route('/api/check_model_status/', methods=['GET'])
def check_model_status():
    status = redis_conn.get('model_status').decode("utf-8")
    return jsonify({'model_status': status})

@app.route('/api/training_progress/', methods=['GET'])
def get_training_progress():
    """
    Gets the current training progress from Redis if available.
    """
    tp_value = redis_conn.get('training_progress')
    if tp_value:
        training_progress = json.loads(tp_value.decode("utf-8"))
        return jsonify(training_progress)
    else:
        return jsonify({'message': 'No training progress available'})
    
@app.route('/api/progress/')
def progress():
    """
    Sends Server-Sent Events with model training progress updates.
    Opens a streaming response and yields training progress JSON messages 
    from Redis until training completes or is interrupted.
    """
    def generate():
        # Initial check for model status
        model_status = redis_conn.get('model_status').decode("utf-8")

        # Loop until training is not in progress
        while model_status == app.config['MODEL_CREATION_STATUS']['IN_PROGRESS']:
            # Getting progress from Redis
            tp_value = redis_conn.get('training_progress')
            if tp_value:
                training_progress = json.loads(tp_value.decode("utf-8"))
                print("Sending SSE data")
                yield f"data: {json.dumps(training_progress)}\n\n"
            time.sleep(1)  # update every second

            # Re-check model status at each iteration
            model_status = redis_conn.get('model_status').decode("utf-8")

        # Send signal to close connection if training has finished or been interrupted
        print("Closing SSE connection")
        message = {
            "event": "close",
            "reason": model_status,
            "other_data": "You can add more data here if needed."
        }
        yield f"data: {json.dumps(message)}\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')


@app.route('/api/model_accuracy/', methods=['GET'])
def get_model_accuracy():
    try:
        model_accuracy_bytes = redis_conn.get('model_accuracy')
        model_accuracy = float(model_accuracy_bytes.decode("utf-8"))
    except:
        model_accuracy = None
    return jsonify({'accuracy': model_accuracy})
    
"""
Builds the machine learning model by calling the Trainer class.

Handles model training, storing training results in Redis.
Sets the appropriate status in Redis to indicate if training completed 
successfully, was interrupted, or failed with an error.
"""
def build_model():
    try:
        _, accuracy = trainer.build_model(stop_training_event)
        # If None is returned, training failed
        if accuracy == None:
            redis_conn.set('model_status', app.config['MODEL_CREATION_STATUS']['ERROR'])
        else:
            redis_conn.set('model_accuracy', accuracy)
            if stop_training_event.is_set():
                redis_conn.set('model_status', app.config['MODEL_CREATION_STATUS']['INTERRUPTED'])
            else:
                redis_conn.set('model_status', app.config['MODEL_CREATION_STATUS']['COMPLETED'])
                # Uncomment, if you want to load new model into memory
                #trainer.load_model()
    finally:
        train_model_lock.release()
        redis_conn.set('training_progress', '{}')  # Reset training progress


if __name__=='__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
