import os

class Config:
    MODEL_FILE = "ai_digits_model.h5"
    MODEL_SCRIPT = "ai_digits.py"
    CUR_FOLDER = "."
    MODEL_FILE = "ai_digits_model.h5"
    MODEL_CREATION_STATUS = {
        'NOT_STARTED': "not_started",
        'IN_PROGRESS': "in_progress",
        'COMPLETED': "completed",
        'INTERRUPTED': "interrupted",
        'ERROR': "error"
    }

class Dev(Config):
    DEBUG = True
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_URL = None
    FRONTEND_URL = "http://localhost:3000"


class Prod(Config):
    DEBUG = False
    REDIS_URL = os.getenv('REDIS_URL')
    FRONTEND_URL = os.getenv('FRONTEND_URL')
