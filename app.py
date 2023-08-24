from flask import Flask, request, jsonify
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

FRONTEND_URL = "http://localhost:3000"
MODEL_FILE = "ai_digits_model.h5"
MODEL_SCRIPT = "ai_digits.py"
CUR_FOLDER = "."
MODEL_CREATION_STATUS = ["not_started", "in_progress", "completed", "interrupted"]
train_model_lock = threading.Lock() #global variable to lock the thread that is training the model, so multiple similar processes won't run at the same time
stop_training_event = threading.Event() # Event that will signal the training process to stop
global_model = None

#Connect to an instance of Redis
redis_url = os.getenv('REDIS_URL')
redis_conn = redis.StrictRedis.from_url(redis_url)
redis_conn.set('training_progress', '{}')
redis_conn.set('model_status', MODEL_CREATION_STATUS[0])
redis_conn.set('model_accuracy', 0)


app = Flask(__name__)
CORS(app, origins=[FRONTEND_URL])

# Initialize the model trainer
trainer = DigitModelTrainer()
# Load model
trainer.load_model()

@app.route("/")
def home():
    return jsonify({"status": "healty"})

@app.route('/api/drawing/', methods=['POST'])
def send_drawing():
    if not os.path.isfile(os.path.join(CUR_FOLDER, MODEL_FILE)):
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
    # Remove preceding 'data:image/png;base64,', decode and convert to PIL image obj
    base64_data = image_data.split(',')[1]
    byte_data = base64.b64decode(base64_data)
    drawing = Image.open(BytesIO(byte_data))
    return drawing

 # Resize and convert to grayscale, convert to numpy array, normalize
def preprocess_drawing(drawing):
    drawing = drawing.resize((28, 28)).convert('L') 
    image_arr = np.array(drawing) / 255.0  # Convert to numpy array and normalize
    return image_arr
    
@app.route('/api/retrain_model/', methods=['GET'])
def retrain_model():
    global train_model_lock, stop_training_event

    # Attempt to acquire lock without blocking. If another thread already has
    # the lock (someone else is training the model at the same time), this will return False immediately, avoiding a race condition
    if not train_model_lock.acquire(blocking=False):
        return jsonify({'message': 'Training is already in progress. Please try again later.'}), 409  # 409 Conflict
    
    stop_training_event.clear()
    # Start the model creation in a separate thread, so response can be sent immediately
    thread = threading.Thread(target=create_model_file)
    thread.start()
    redis_conn.set('model_status', MODEL_CREATION_STATUS[1])
    return jsonify({'message': 'Model creation started'}), 202 #202 Accepted status code is used to indicate that a request \
                                                    #has been accepted for processing, but the processing has not yet been completed

@app.route('/api/stop_training/', methods=['GET'])
def stop_training():
    print("Gonna stop training...")
    global stop_training_event
    stop_training_event.set()
    return jsonify({'message': 'Stop signal sent.'})
    
@app.route('/api/check_model_status/', methods=['GET'])
def check_model_status():
    status = redis_conn.get('model_status').decode("utf-8")
    return jsonify({'model_status': status})

@app.route('/api/training_progress/', methods=['GET'])
def get_training_progress():
    tp_value = redis_conn.get('training_progress')
    if tp_value:
        training_progress = json.loads(tp_value.decode("utf-8"))
        return jsonify(training_progress)
    else:
        return jsonify({'message': 'No training progress available'})

@app.route('/api/model_accuracy/', methods=['GET'])
def get_model_accuracy():
    try:
        model_accuracy_bytes = redis_conn.get('model_accuracy')
        model_accuracy = float(model_accuracy_bytes.decode("utf-8"))
    except:
        model_accuracy = None
    return jsonify({'accuracy': model_accuracy})
    
def create_model_file():
    try:
        _, accuracy = trainer.build_model(stop_training_event)
        redis_conn.set('model_accuracy', accuracy)
        if stop_training_event.is_set():
            redis_conn.set('model_status', MODEL_CREATION_STATUS[3])
        else:
            redis_conn.set('model_status', MODEL_CREATION_STATUS[2])
            # Load new model into memory after a successfull creation
            trainer.load_model()
    finally:
        train_model_lock.release()


if __name__=='__main__':
    app.run(debug=True)