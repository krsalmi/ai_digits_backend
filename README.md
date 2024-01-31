# Backend for the AI Digit Recognition Web App
Backend for a web application, which uses machine learning to recognize handwritten digits. The server is a Flask app with Redis used for memory caching, the model used for digit recognition is a Convolutional Neural Network, built using TensorFlow's Keras library.
Provides API endpoints for model retraining, digit prediction, progress tracking, and model status checking.
Retrained models will not be saved into memory, but this can be turned on in build_model() if needed. You can check out the live site on
[https://digitrecognition-ai.onrender.com](https://digitrecognition-ai.onrender.com), which is hosted on Render.
The code for the frontend can be viewed [here](https://github.com/krsalmi/ai_digits_frontend.git).

## Model Architecture and Training Process
The `DigitModelTrainer` class handles the process of building and training a Convolutional Neural Network (CNN) for digit recognition using the MNIST dataset.   
<img src="imgs_for_readme/mnist-3.0.1.png" alt="MNIST example image" width="200" height="200">  
Example images from the MNIST dataset  
  
Key steps involved:

### Dataset Preparation
- **Data Source**: The MNIST dataset, a large collection of handwritten digits.
- **Data Preprocessing**: The dataset is loaded and preprocessed to conform to the model's input requirements. This includes reshaping the images to add a channel dimension, normalizing pixel values, and one-hot encoding the labels.

### Model Building
- **Architecture**: A CNN model is constructed with convolutional layers, pooling layers, a flattening and a hidden layer, and a dropout layer to mitigate overfitting.
- **Optimization**: The Adam optimizer with an exponential decay learning rate schedule is employed. The model is compiled with categorical cross-entropy loss and accuracy as the evaluation metric.

### Training Process
- **Dataset Division**: The MNIST dataset is partitioned into training and testing sets.
- **Training**: The model undergoes training on the training data across multiple epochs. The process includes provisions for early stopping, along with logging of training progress, memory usage, and time durations.
- **Accuracy Evaluation**: Post-training, the model is evaluated on the test set to determine its accuracy.

### Prediction
- **Inference**: The trained model is capable of predicting digits from new 28x28 pixel images, providing both the predicted digit and the confidence level of the prediction.

## Approach to implementing machine learning online, and things I learned
The live site for this project is hosted on a Render instance with 1 CPU and 2 GB of storage. This is not very efficent, and because of that, I have set the training epochs (complete passes through the entire training dataset to improve the ML model) to 6. This way, the training will take around 8 minutes from start to finish, which is quite long and still will not achieve very high accuracy scores during the testing phase. However, hosting on a GPU instance was too expensive in my opinion, because my goal was just to give the users the experience of live training of a model, not to actually use that model online. Because of this, the model which will be uploaded into memory is one I pretrained on my M1 chip Mac, with the number of epochs set to 20. The model is in **ai_digits_model.h5** and it's accuracy is close 98.91%, which is very good.  

I found the CNN architecture and configuration, one I was happy with, through manual experimentation. During the building and testing of the model, I was surprise to notice that adding more than 1 hidden layer actually decreased the achieved accuracy, and the same happened if there were more than 128 nodes in the hidden layer. Having 2 pairs of MaxPooling and Conv2D layers worked well when there were more filters in the first pair than in the second. I wanted to keep the dropout big (at 50%) to avoid overfitting. The optimization algorithmn ("adam" in this case) has a decayed learning rate the longer the training goes on. This I found worked because I noticed how initially there was not much progress for the achieved accuracy during the later training epochs. 

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
