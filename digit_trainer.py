from tensorflow import keras
import redis
import json
import os
import numpy as np
import logging
import psutil
import datetime
import time
from config import Dev, Prod


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
EPOCH_NUM = 3

"""
Stops model training when a stop event is triggered. 

Sets a class property `training_stopped` to True when the stop event is triggered, 
allowing other callbacks to check if training was stopped.
"""
class StopTrainingCallback(keras.callbacks.Callback):
    training_stopped = False

    def __init__(self, stop_event):
        super().__init__()
        self.stop_event = stop_event

    def on_batch_end(self, batch, logs=None):
        if self.stop_event.is_set():
            self.model.stop_training = True
            StopTrainingCallback.training_stopped = True



"""
Callback that updates Redis with training progress after each batch and epoch.

Tracks and updates the latest training progress in a Redis hash. 
Progress contains the current epoch, batch percentage 
complete, and latest accuracy. Updates are made after each batch and epoch 
during model training.
"""
class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, redis_conn, batch_frequency=20):
        super().__init__()
        self.redis_conn = redis_conn
        self.batch_frequency = batch_frequency
        self.batch_count = 0
        self.latest_progress = {'max_epochs': EPOCH_NUM}

    def update_redis(self):
        self.redis_conn.set('training_progress', json.dumps(self.latest_progress))

    def on_batch_end(self, batch, logs=None):
        # Check if StopTrainingCallback flag is set, and if so, skip progress update.
        if StopTrainingCallback.training_stopped:
            return

        self.batch_count += 1
        if self.batch_count % self.batch_frequency == 0:
            self.latest_progress['percentage'] = self.batch_count / self.params['steps'] * 100
            self.update_redis()

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_count = 0
        self.latest_progress['percentage'] = 0
        self.update_redis()

    def on_epoch_end(self, epoch, logs=None):
        # Check if StopTrainingCallback flag is set, and if so, skip progress update.
        if StopTrainingCallback.training_stopped:
            return
        
        accuracy = logs.get('accuracy')
        self.latest_progress['accuracy'] = "{:.2f}".format(accuracy * 100)
        self.latest_progress['epoch'] = epoch + 1
        self.update_redis()


"""
A class for training a digit recognition model using a convolutional neural network (CNN).

This class is responsible for initializing and training a neural network model for
recognizing handwritten digits. It utilizes the MNIST dataset for training and testing.
The class provides functionalities to load and preprocess the dataset, build, train, and 
evaluate the CNN model. It also includes methods to predict digits from input images.
"""
class DigitModelTrainer:
    NUM_CLASSES = 10
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    CHANNEL_DIMENSION = 1
    REDIS_URL = os.getenv('REDIS_URL')
    BATCH_SIZE = 128
    

    def __init__(self, env='development'):
        self.load_config(env)
        self.initialize_redis()
        self.model = None

    def load_config(self, env):
        if env == 'development':
            self.config = Dev()
        else:
            self.config = Prod()

    def initialize_redis(self):
        if self.config.REDIS_URL:
            self.redis_conn = redis.StrictRedis.from_url(self.config.REDIS_URL)
        else:
            self.redis_conn = redis.StrictRedis(host=self.config.REDIS_URL, port=6379, db=0)

    def load_model(self):
        if os.path.isfile(self.config.MODEL_FILE):
            self.model = keras.models.load_model(self.config.MODEL_FILE)
        else:
            print("Model file does not exist. Please create the model first.")

    def load_dataset(self):
        try:
            # Load MNIST data
            (self.X_train, self.y_train), (self.X_test, self.y_test) = keras.datasets.mnist.load_data()
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e

    """
    Builds the CNN model for digit recognition, trains it, and returns the trained model and its accuracy.
    
    Loads the MNIST dataset, preprocesses the data, initializes a CNN model architecture, 
    trains the model while allowing external stopping via an event, and returns the trained model and its accuracy.
    
    Handles logging training progress, memory usage, durations etc. 
    """
    def build_model(self, stop_training_event):
        logging.info("Starting to build the model.")
        logging.info(f"Current memory usage: {psutil.virtual_memory().percent}%")
        logging.info(f"Current CPU usage: {psutil.cpu_percent()}%")

        start_time = time.time()

        #Load dataset upon starting, just in case it is needed
        self.load_dataset()

        # Reset the flag in stopTrainingCallback
        StopTrainingCallback.training_stopped = False  # Reset the flag

        self.prepare_data_for_model()
        self.initialize_model()
        model, accuracy = self.train_model(stop_training_event, batch_size=self.BATCH_SIZE)


        end_time = time.time()
        elapsed_time = end_time - start_time

        logging.info(f"Training over, took: {elapsed_time:.2f} seconds.")
        return model, accuracy
        
    """
    Prepares the MNIST image data for training and testing the CNN model. 
    
    Reshapes the loaded MNIST image data to add a channel dimension, 
    normalizes the pixel values, and one-hot encodes the labels.
    
    Handles logging of memory usage and durations.
    """
    def prepare_data_for_model(self):
        logging.info(f"Preparing data at {datetime.datetime.now()}")
        try:
            num_training_images = self.X_train.shape[0]
            num_testing_images = self.X_test.shape[0]
            # Reshape the data and add channel dimension
            self.X_train = self.X_train.reshape(num_training_images, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL_DIMENSION)
            self.X_test= self.X_test.reshape(num_testing_images, self.IMG_HEIGHT, self.IMG_WIDTH, self.CHANNEL_DIMENSION)

            # Normalize pixel values
            self.X_train, self.X_test = self.X_train / 255.0, self.X_test / 255.0

            # Create One-hot encode labels.
            self.y_train = keras.utils.to_categorical(self.y_train, self.NUM_CLASSES)
            self.y_test = keras.utils.to_categorical(self.y_test, self.NUM_CLASSES)

            logging.info(f"Data prepared successfully at {datetime.datetime.now()}")
            logging.info(f"Memory usage after data preparation: {psutil.virtual_memory().percent}%")
            logging.info(f"CPU usage after data preparation: {psutil.cpu_percent()}%")
            

        except Exception as e:
            logging.error(f"Error occured while preparing the data for the model at {datetime.datetime.now()}: {e}")
            raise e
    
    """
    Initializes the convolutional neural network model for digit classification.
    
    Creates a sequential CNN model with convolutional, pooling, flatten, dense and 
    dropout layers. Configures the Adam optimizer with an exponential decay learning 
    rate schedule. Compiles the model with categorical cross entropy loss and accuracy 
    metric.
    
    Handles logging of memory and CPU usage after model initialization.
    """
    def initialize_model(self):
        logging.info(f"Initializing model at {datetime.datetime.now()}")
        # Create a convolutional neural network
        try:
            self.model = keras.models.Sequential([
                keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10, activation='softmax')
            ])

            # Learning rate scheduling. The optimization algorithmn ("adam" in this case)
            # will have a decayed learning rate the longer the training goes on
            initial_learning_rate = 0.001
            decay_steps = 400
            decay_rate = 0.93

            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=decay_steps,
                decay_rate=decay_rate,
                staircase=True)

            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

            self.model.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            logging.info(f"Model initialized successfully at {datetime.datetime.now()}")
            logging.info(f"Memory usage after model initialization: {psutil.virtual_memory().percent}%")
            logging.info(f"CPU usage after model initialization: {psutil.cpu_percent()}%")

        except Exception as e:
            logging.error(f"Error occured while initializing the model at {datetime.datetime.now()}: {e}")
            raise e
        
    """
    Trains the CNN model on the training data.
    
    Fits the model on the training data for a number of epochs, with a batch size. 
    Uses a callback to check if training should stop early. Saves training progress info to Redis.
    Evaluates the model on the test data after training.
    
    Returns a tuple with the trained model and accuracy on the test set.
    """
    def train_model(self, stop_training_event, batch_size=32):
        logging.info(f"Starting model training at {datetime.datetime.now()}")
        try:
            # New instance of callback class, which will check the signal (always when a batch ends)
            stop_training_callback = StopTrainingCallback(stop_training_event)

            # Callback that saves progress info when batches and epochs end
            progress_callback = ProgressCallback(self.redis_conn)

            self.model.fit(self.X_train, self.y_train, epochs=EPOCH_NUM, batch_size=batch_size, callbacks=[progress_callback, stop_training_callback])

            # Evaluate neural network performance
            _, accuracy = self.model.evaluate(self.X_test,  self.y_test, verbose=2, batch_size=batch_size)

            # Uncomment if you want to save new model into file
            #self.model.save(self.config.MODEL_FILE)

            logging.info(f"Model trained successfully at {datetime.datetime.now()}")
            logging.info(f"Memory usage after model training: {psutil.virtual_memory().percent}%")
            logging.info(f"CPU usage after model training: {psutil.cpu_percent()}%")

            return self.model, accuracy
        except Exception as e:
            logging.error(f"Error occured while training the model at {datetime.datetime.now()}: {e}")
            return None, None
    
    
    # Analyze drawing and predict digit
    # returns the index of the maximum value in the provided array, which has probabilities for each digit
    """
    Analyze the drawing and predict the digit.
    
    Takes a 28x28 pixel image as a numpy array, reshapes it for the model input shape, runs inference, 
    and returns the predicted digit (0-9) with the confidence percentage.
    
    Returns:
        predicted_digit (int): The predicted digit (0-9).
        confidence_percentage (float): The confidence percentage that the predicted digit is correct.
    """
    def predict_digit(self, image_arr):
        if self.model is None:
            return (None, None)
        try:
            input_data = image_arr.reshape(1, 28, 28, 1)  # Reshape for the model
            prediction_array = self.model.predict(input_data)
            predicted_digit = int(np.argmax(prediction_array))  # Get the index of the maximum value which is our digit and convert to native Python int
            # Convert the max probability to percentage (of confidence)
            # Assuming prediction[0] exists here is ok, keras returns (batch_size, num_classes) --> batch_size is 1, because we're analyzing 1 image
            confidence_percentage = float(prediction_array[0][predicted_digit] * 100) 
            print(f"Predicted digit: {predicted_digit}, confidence {confidence_percentage}%")
            return predicted_digit, confidence_percentage
        except Exception as e:
            logging.error(f"An error occured while predicting digit: {e}")
            raise e
