# import logging
# import traceback
# from flask import Flask, jsonify
# from flask_cors import CORS
# from routes.api import api_bp
# from utils.database import init_db
# from config import Config
# # Try importing with a relative path if 'utils' is a package in the backend directory
# # from .utils.media_processing import process_media_file  # Use relative import if running as a module

# # If the above does not work, or if running as a script, use:
# # from utils.media_processing import process_media_file


# app = Flask(__name__)
# app.config.from_object(Config)
# CORS(app)

# # Init DB
# init_db(app)

# # Add to your app.py
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# @api_bp.route('/upload', methods=['POST'])
# def upload_file():
#     logger.info("=== FILE UPLOAD STARTED ===")
#     try:
#         # Extract file from request
#         if 'file' not in flask.request.files:
#             return jsonify({'error': 'No file part in the request'}), 400
#         file = flask.request.files['file']
#         if file.filename == '':
#             return jsonify({'error': 'No selected file'}), 400
#         filename = file.filename
#         file_path = f"/tmp/{filename}"
#         file.save(file_path)
#         result = process_media_file(file_path, filename)
#         return jsonify(result)
#     except Exception as e:
#         logger.error(f"PROCESSING FAILED: {str(e)}")
#         logger.error(f"Full traceback: {traceback.format_exc()}")
#         return jsonify({'error': f'Processing failed: {str(e)}'}), 500

# # Register blueprints
# app.register_blueprint(api_bp, url_prefix='/api')

# @app.route('/')
# def home():
#     return jsonify({'status': 'healthy', 'msg': 'API running'})

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)




# app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import logging
# from werkzeug.utils import secure_filename

# # ✅ FIXED: Use absolute imports (remove the dots)
# from utils.media_processing import process_media_file
# from models.audio_model import AudioEmotionModel
# from models.video_model import VideoEmotionModel
# from utils.database import init_db

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# CORS(app)

# # Configuration
# app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['MONGODB_URI'] = 'mongodb://localhost:27017/emotion_recognition'

# # Ensure upload directory exists
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Initialize database
# init_db(app)

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     try:
#         return jsonify({
#             'status': 'healthy',
#             'message': 'Multimodal Emotion Recognition API is running',
#             'models_loaded': {
#                 'audio': True,
#                 'video': True,
#                 'text': True
#             }
#         })
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/api/upload', methods=['POST'])
# def upload_file():
#     logger.info("=== FILE UPLOAD STARTED ===")
    
#     try:
#         # Check if file is in request
#         if 'file' not in request.files:
#             logger.error("No 'file' key in request.files")
#             return jsonify({'error': 'No file provided'}), 400
        
#         file = request.files['file']
#         if file.filename == '':
#             logger.error("Empty filename")
#             return jsonify({'error': 'No file selected'}), 400
        
#         # Log file details
#         logger.info(f"File received: {file.filename}")
        
#         # Validate file type
#         allowed_extensions = {'wav', 'mp3', 'mp4', 'avi', 'mov', 'mkv'}
#         file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
#         if file_ext not in allowed_extensions:
#             return jsonify({'error': f'Unsupported file type: {file_ext}'}), 400
        
#         # Save file
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
#         logger.info(f"File saved to: {file_path}")
        
#         # Process file
#         logger.info("Starting multimodal processing...")
#         result = process_media_file(file_path, filename)
#         logger.info("Processing completed successfully!")
        
#         return jsonify(result)
        
#     except Exception as e:
#         logger.error(f"UPLOAD FAILED: {str(e)}")
#         logger.error(f"Error type: {type(e).__name__}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
        
#         return jsonify({
#             'error': f'Failed to process file: {str(e)}',
#             'error_type': type(e).__name__
#         }), 500

# if __name__ == '__main__':
#     logger.info("Starting Multimodal Emotion Recognition Backend...")
#     app.run(debug=True, host='0.0.0.0', port=5000)



# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename

# Import our modules
from utils.media_processing import process_media_file
from utils.database import init_db

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MONGODB_URI'] = 'mongodb://localhost:27017/emotion_recognition'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
init_db(app)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Multimodal Emotion Recognition API is running',
        'version': '1.0.0'
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """File upload and processing endpoint"""
    logger.info("=== FILE UPLOAD STARTED ===")
    
    try:
        # Check if file is in request
        if 'file' not in request.files:
            logger.error("No 'file' key in request.files")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        # Log file details
        logger.info(f"File received: {file.filename}")
        
        # Validate file type
        allowed_extensions = {'wav', 'mp3', 'mp4', 'avi', 'mov', 'mkv'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({'error': f'Unsupported file type: {file_ext}'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f"File saved to: {file_path}")
        
        # Process file
        logger.info("Starting multimodal processing...")
        result = process_media_file(file_path, filename)
        logger.info("Processing completed successfully!")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"UPLOAD FAILED: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'error': f'Failed to process file: {str(e)}'
        }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Multimodal Emotion Recognition Backend...")
    app.run(debug=True, host='0.0.0.0', port=5000)
