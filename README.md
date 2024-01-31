# Flask App for AI Digit Recognition

## Introduction
Backend for a web application, which uses machine learning to recognize handwritten digits. The server is a Flask app with Redis used for memory caching, the model used for digit recognition is a Convolutional Neural Network, built using TensorFlow's Keras library.
Provides API endpoints for model training, digit prediction, progress tracking, and model status checking.
Retrained model will not be saved into memory, but this can be turned on in build_model() if needed. You can check out the live site on
[https://digitrecognition-ai.onrender.com](https://digitrecognition-ai.onrender.com), which is hosted on Render.
The code for the frontend can be viewed [here](https://github.com/krsalmi/ai_digits_frontend.git).

## Details on the Convolutional Neural Network (CNN)
The Convolutional Neural Network (CNN) in **digit_trainer.py** is designed for recognizing handwritten digits, specifically trained on the MNIST dataset. It consists of a sequential model architecture built using TensorFlow's Keras API, featuring two sets of convolutional layers (each followed by max-pooling layers), a flattening layer, a dense layer with 128 neurons, and a dropout layer for regularization. The convolutional layers use 32 and 64 filters respectively, each with a kernel size of 5x5, and are activated by the ReLU function. The model's output layer is a dense layer with 10 neurons (representing the 10 digit classes), using the softmax activation function for multi-class classification. This model structure is optimized using the Adam optimizer with a learning rate that decays exponentially, and it is compiled with categorical cross-entropy as the loss function, focusing on accuracy as the performance metric. This setup makes it well-suited for high-accuracy digit recognition tasks.

## Other notable info
### SSE
The /api/progress/ endpoint in this Flask application streams model training progress updates as Server-Sent Events (SSE). It uses a generator function to continuously fetch training progress data from Redis and sends this data as JSON-formatted messages to the client. The streaming continues until the model training is either completed or interrupted, indicated by the model status in Redis. On completion or interruption, the function sends a closing message and terminates the SSE connection. This real-time update mechanism is efficient for monitoring long-running model training processes.

## Setup Instructions

### Requirements
- Python (recommended version:) 3.10.2
- Flask
- Redis
- TensorFlow
- PIL (Python Imaging Library)

### Installation
1. Clone the repository.
2. Create a python environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
- Modify `config.py` to adjust development (`Dev`) and production (`Prod`) settings as needed.
- Set the necessary environment variables for production, including `REDIS_URL` and `FRONTEND_URL`.

### Running the Service
- To start the service in development mode:
  ```bash
  export FLASK_ENV=development
  python app.py
  ```

## Usage

### API Endpoints
- `GET /`: Health check endpoint.
- `POST /api/drawing/`: Predicts the digit from a given drawing.
- `POST /api/retrain_model/`: Starts the retraining of the model.
- `POST /api/stop_training/`: Stops the ongoing training process.
- `GET /api/check_model_status/`: Checks the current status of the model.
- `GET /api/training_progress/`: Gets the training progress.
- `GET /api/model_accuracy/`: Retrieves the accuracy of the trained model.

## Components

### app.py
This is the main Flask application file. It defines API routes and integrates the digit recognition model with Redis for session storage and training progress tracking.

### digit_trainer.py
Contains the `DigitModelTrainer` class, responsible for training the digit recognition model using TensorFlow. It includes methods for data preparation, model building, training, and prediction.

### config.py
Defines configuration classes for different environments (development and production), including settings for the model file, Redis, and frontend URL.
