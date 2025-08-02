import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'supersecret'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'data', 'temp')
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024

    MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/emotion_recognition')

    # Allowed file types
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'avi', 'mov', 'mkv'}

    @staticmethod
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
