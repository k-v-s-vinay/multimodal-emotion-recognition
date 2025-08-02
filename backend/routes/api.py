from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import tempfile
import json
from models.audio_model import AudioEmotionModel
from models.text_model import TextEmotionModel
from models.video_model import VideoEmotionModel
from models.fusion_model import MultimodalFusion
from utils.database import save_analysis_result, get_analysis_result
from config import Config

api_bp = Blueprint('api', __name__)

# Initialize models (in production, consider using a model manager)
audio_model = AudioEmotionModel()
text_model = TextEmotionModel()
video_model = VideoEmotionModel()
fusion_model = MultimodalFusion()

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process media file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not Config.allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        
        # Process the file
        result = process_media_file(file_path, filename)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_media_file(file_path, filename):
    """Process uploaded media file"""
    file_extension = filename.lower().split('.')[-1]
    
    # Determine file type and process accordingly
    if file_extension in ['wav', 'mp3']:
        # Audio only
        audio_results = audio_model.process_audio_file(file_path)
        text_results = text_model.process_conversation_text(audio_results['speaker_segments'])
        video_results = {}
        
    elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
        # Video (contains audio and video)
        audio_results = audio_model.process_audio_file(file_path)
        text_results = text_model.process_conversation_text(audio_results['speaker_segments'])
        video_results = video_model.process_video_file(file_path)
        
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    # Fuse multimodal results
    fused_results = fusion_model.process_multimodal_data(
        audio_results, text_results, video_results
    )
    
    # Prepare final result
    final_result = {
        'filename': filename,
        'file_type': file_extension,
        'processing_results': {
            'audio': audio_results,
            'text': text_results,
            'video': video_results,
            'fused': fused_results
        },
        'summary': fused_results.get('summary', {}),
        'segments': fused_results.get('segments', [])
    }
    
    # Save to database
    result_id = save_analysis_result(final_result)
    final_result['analysis_id'] = result_id
    
    return final_result

@api_bp.route('/analyze/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Retrieve previous analysis result"""
    try:
        result = get_analysis_result(analysis_id)
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Analysis not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'audio': audio_model is not None,
            'text': text_model is not None,
            'video': video_model is not None,
            'fusion': fusion_model is not None
        }
    })

@api_bp.route('/emotions', methods=['GET'])
def get_emotion_labels():
    """Get available emotion labels"""
    return jsonify({
        'emotion_labels': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
        'supported_file_types': list(Config.ALLOWED_EXTENSIONS)
    })
