from tensorflow import keras
import redis
import json
import os
import numpy as np
import logging
import psutil
import datetime
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class StopTrainingCallback(keras.callbacks.Callback):
    training_stopped = False

    def __init__(self, stop_event):
        super().__init__()
        self.stop_event = stop_event

    def on_batch_end(self, batch, logs=None):
        if self.stop_event.is_set():
            print("Inside StopTrainingCallback, stop was received")
            self.model.stop_training = True
            StopTrainingCallback.training_stopped = True


class DigitModelTrainer:
    NUM_CLASSES = 10
    IMG_HEIGHT = 28
    IMG_WIDTH = 28
    CHANNEL_DIMENSION = 1
    MODEL_FILE = "ai_digits_model.h5"
    REDIS_URL = os.getenv('REDIS_URL')
    EPOCH_NUM = 7
    

    def __init__(self):
        # Connect to Redis instance
        #self.redis_conn = redis.StrictRedis.from_url(self.REDIS_URL)
        self.redis_conn = redis.StrictRedis(host='localhost', port=6379, db=0)

        self.model = None

    def load_model(self):
        if os.path.isfile(self.MODEL_FILE):
            self.model = keras.models.load_model(self.MODEL_FILE)
        else:
            print("Model file does not exist. Please create the model first.")

    def load_dataset(self):
        try:
            # Load MNIST data
            (self.X_train, self.y_train), (self.X_test, self.y_test) = keras.datasets.mnist.load_data()
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise e


    def save_epoch_details(self, epoch, logs=None):
        logging.info(f"Saving details for epoch {epoch}.")

        if StopTrainingCallback.training_stopped:
            print("About to save epoch details. Training has stopped.")
            return

        print("saving epoch details")
        logs = logs or {}
        updated_progress = {
            'accuracy': logs.get('accuracy'),
            'epoch': epoch + 1
        }
        self.redis_conn.set('training_progress', json.dumps(updated_progress))
        logging.info(f"Details saved for epoch {epoch}.")

    def build_model(self, stop_training_event):
        logging.info("Starting to build the model.")
        logging.info(f"Current memory usage: {psutil.virtual_memory().percent}%")
        logging.info(f"Current CPU usage: {psutil.cpu_percent()}%")

        start_time = time.time()

        # Reset the flag in stopTrainingCallback
        StopTrainingCallback.training_stopped = False  # Reset the flag

        self.prepare_data_for_model()
        self.initialize_model()
        model, accuracy = self.train_model(stop_training_event, batch_size=64)


        end_time = time.time()
        elapsed_time = end_time - start_time

        logging.info(f"Training over, took: {elapsed_time:.2f} seconds.")
        return model, accuracy
        
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
    
    def initialize_model(self):
        logging.info(f"Initializing model at {datetime.datetime.now()}")
        # Create a convolutional neural network
        try:
            self.model = keras.models.Sequential([
                keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(64, kernel_size=(5, 5), activation='relu'),
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
        
    def train_model(self, stop_training_event, batch_size=32):
        logging.info(f"Starting model training at {datetime.datetime.now()}")
        try:
            # New instance of callback class, which will check the signal (always when a batch ends)
            stop_training_callback = StopTrainingCallback(stop_training_event)

            # Callback which will run 'save_epoch_detail' on each epoch end
            epoch_callback = keras.callbacks.LambdaCallback(on_epoch_end=self.save_epoch_details)

            self.model.fit(self.X_train, self.y_train, epochs=EPOCH_NUM, batch_size=batch_size, callbacks=[epoch_callback, stop_training_callback])

            # Evaluate neural network performance
            _, accuracy = self.model.evaluate(self.X_test,  self.y_test, verbose=2, batch_size=batch_size)

            self.model.save(self.MODEL_FILE)

            logging.info(f"Model trained successfully at {datetime.datetime.now()}")
            logging.info(f"Memory usage after model training: {psutil.virtual_memory().percent}%")
            logging.info(f"CPU usage after model training: {psutil.cpu_percent()}%")

            return self.model, accuracy
        except Exception as e:
            logging.error(f"Error occured while training the model at {datetime.datetime.now()}: {e}")
    
    
    # Analyze drawing and predict digit
    # returns the index of the maximum value in the provided array, which has probabilities for each digit
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
