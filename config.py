import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'healthsync-secret-key-2024'
    DEBUG = True
    
    # Paths
    DATA_PATH = 'data/'
    MODEL_PATH = 'models/'
    UPLOAD_FOLDER = 'uploads/'
    
    # Application settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    NUTRITION_DB_FILE = 'data/nutrition.csv'