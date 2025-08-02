import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

class MultimodalFusion:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fusion_model = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
    def align_speaker_data(self, audio_results, text_results, video_results):
        """Align data from different modalities by speaker and time"""
        aligned_data = []
        
        if not audio_results.get('speaker_segments'):
            return aligned_data
        
        for audio_segment in audio_results['speaker_segments']:
            speaker = audio_segment['speaker']
            start_time = audio_segment['start_time']
            end_time = audio_segment['end_time']
            
            # Find corresponding text data
            text_data = None
            for text_segment in text_results:
                if (text_segment['speaker'] == speaker and 
                    abs(text_segment['start_time'] - start_time) < 1.0):  # 1 second tolerance
                    text_data = text_segment
                    break
            
            # Find corresponding video data (person closest in time)
            video_data = None
            if video_results.get('person_emotions'):
                best_match = None
                min_time_diff = float('inf')
                
                for person in video_results['person_emotions']:
                    for emotion_entry in person['emotion_timeline']:
                        time_diff = abs(emotion_entry['timestamp'] - (start_time + end_time) / 2)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            best_match = {
                                'person_id': person['person_id'],
                                'emotions': emotion_entry['emotions'],
                                'timestamp': emotion_entry['timestamp']
                            }
                
                if best_match and min_time_diff < 5.0:  # 5 second tolerance
                    video_data = best_match
            
            # Combine data
            aligned_segment = {
                'speaker': speaker,
                'start_time': start_time,
                'end_time': end_time,
                'text': audio_segment.get('text', ''),
                'audio_emotions': audio_segment.get('audio_emotions', {}),
                'text_emotions': text_data.get('basic_emotions', {}) if text_data else {},
                'video_emotions': video_data.get('emotions', {}) if video_data else {},
                'person_id': video_data.get('person_id') if video_data else None
            }
            
            aligned_data.append(aligned_segment)
        
        return aligned_data
    
    def create_feature_vector(self, aligned_segment):
        """Create feature vector from aligned multimodal data"""
        features = []
        
        # Audio emotion features
        audio_emotions = aligned_segment.get('audio_emotions', {})
        for emotion in self.emotion_labels:
            features.append(audio_emotions.get(emotion, 0.0))
        
        # Text emotion features
        text_emotions = aligned_segment.get('text_emotions', {})
        for emotion in self.emotion_labels:
            # Map text emotion labels to standard labels if needed
            text_score = 0.0
            if emotion in text_emotions:
                text_score = text_emotions[emotion]
            elif emotion == 'happy' and 'joy' in text_emotions:
                text_score = text_emotions['joy']
            elif emotion == 'angry' and 'anger' in text_emotions:
                text_score = text_emotions['anger']
            features.append(text_score)
        
        # Video emotion features
        video_emotions = aligned_segment.get('video_emotions', {})
        for emotion in self.emotion_labels:
            features.append(video_emotions.get(emotion, 0.0))
        
        # Additional features
        features.append(len(aligned_segment.get('text', '')))  # Text length
        features.append(aligned_segment.get('end_time', 0) - aligned_segment.get('start_time', 0))  # Duration
        features.append(1.0 if aligned_segment.get('person_id') is not None else 0.0)  # Has video data
        
        return np.array(features)
    
    def weighted_fusion(self, aligned_segment, weights=None):
        """Perform weighted fusion of emotion predictions"""
        if weights is None:
            weights = {'audio': 0.4, 'text': 0.3, 'video': 0.3}
        
        audio_emotions = aligned_segment.get('audio_emotions', {})
        text_emotions = aligned_segment.get('text_emotions', {})
        video_emotions = aligned_segment.get('video_emotions', {})
        
        # Normalize weights based on available modalities
        available_modalities = []
        if audio_emotions:
            available_modalities.append('audio')
        if text_emotions:
            available_modalities.append('text')
        if video_emotions:
            available_modalities.append('video')
        
        if not available_modalities:
            return {'neutral': 1.0}
        
        # Adjust weights
        total_weight = sum(weights[mod] for mod in available_modalities)
        normalized_weights = {mod: weights[mod]/total_weight for mod in available_modalities}
        
        # Fuse emotions
        fused_emotions = {}
        for emotion in self.emotion_labels:
            score = 0.0
            
            if 'audio' in available_modalities:
                score += normalized_weights['audio'] * audio_emotions.get(emotion, 0.0)
            
            if 'text' in available_modalities:
                # Handle emotion label mapping
                text_score = 0.0
                if emotion in text_emotions:
                    text_score = text_emotions[emotion]
                elif emotion == 'happy' and 'joy' in text_emotions:
                    text_score = text_emotions['joy']
                elif emotion == 'angry' and 'anger' in text_emotions:
                    text_score = text_emotions['anger']
                score += normalized_weights['text'] * text_score
            
            if 'video' in available_modalities:
                score += normalized_weights['video'] * video_emotions.get(emotion, 0.0)
            
            fused_emotions[emotion] = score
        
        # Normalize to ensure probabilities sum to 1
        total = sum(fused_emotions.values())
        if total > 0:
            fused_emotions = {k: v/total for k, v in fused_emotions.items()}
        
        return fused_emotions
    
    def attention_fusion(self, aligned_segment):
        """Perform attention-based fusion"""
        # Simple attention mechanism based on confidence scores
        audio_emotions = aligned_segment.get('audio_emotions', {})
        text_emotions = aligned_segment.get('text_emotions', {})
        video_emotions = aligned_segment.get('video_emotions', {})
        
        # Calculate confidence for each modality (max probability)
        audio_conf = max(audio_emotions.values()) if audio_emotions else 0.0
        text_conf = max(text_emotions.values()) if text_emotions else 0.0
        video_conf = max(video_emotions.values()) if video_emotions else 0.0
        
        # Softmax attention weights
        confs = np.array([audio_conf, text_conf, video_conf])
        if confs.sum() > 0:
            attention_weights = np.exp(confs) / np.sum(np.exp(confs))
        else:
            attention_weights = np.array([1/3, 1/3, 1/3])
        
        # Apply attention weights
        weights = {
            'audio': attention_weights[0],
            'text': attention_weights[1], 
            'video': attention_weights[2]
        }
        
        return self.weighted_fusion(aligned_segment, weights)
    
    def process_multimodal_data(self, audio_results, text_results, video_results):
        """Main function to process and fuse multimodal data"""
        # Align data from different modalities
        aligned_data = self.align_speaker_data(audio_results, text_results, video_results)
        
        if not aligned_data:
            return {
                'error': 'No aligned data found',
                'individual_results': {
                    'audio': audio_results,
                    'text': text_results,
                    'video': video_results
                }
            }
        
        # Process each aligned segment
        fused_results = []
        for segment in aligned_data:
            # Weighted fusion
            weighted_emotions = self.weighted_fusion(segment)
            
            # Attention-based fusion
            attention_emotions = self.attention_fusion(segment)
            
            # Combine results
            result = {
                'speaker': segment['speaker'],
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'text': segment['text'],
                'person_id': segment.get('person_id'),
                'individual_predictions': {
                    'audio': segment.get('audio_emotions', {}),
                    'text': segment.get('text_emotions', {}),
                    'video': segment.get('video_emotions', {})
                },
                'fused_predictions': {
                    'weighted': weighted_emotions,
                    'attention': attention_emotions
                },
                'final_emotion': max(weighted_emotions, key=weighted_emotions.get),
                'confidence': max(weighted_emotions.values())
            }
            
            fused_results.append(result)
        
        # Generate conversation summary
        summary = self.generate_conversation_summary(fused_results)
        
        return {
            'segments': fused_results,
            'summary': summary,
            'total_segments': len(fused_results),
            'processing_status': 'success'
        }
    
    def generate_conversation_summary(self, fused_results):
        """Generate overall conversation summary"""
        if not fused_results:
            return {}
        
        # Aggregate emotions
        all_emotions = {emotion: [] for emotion in self.emotion_labels}
        speaker_emotions = {}
        
        for result in fused_results:
            speaker = result['speaker']
            if speaker not in speaker_emotions:
                speaker_emotions[speaker] = {emotion: [] for emotion in self.emotion_labels}
            
            weighted_emotions = result['fused_predictions']['weighted']
            for emotion, score in weighted_emotions.items():
                if emotion in all_emotions:
                    all_emotions[emotion].append(score)
                    speaker_emotions[speaker][emotion].append(score)
        
        # Calculate averages
        emotion_averages = {emotion: np.mean(scores) if scores else 0.0 
                           for emotion, scores in all_emotions.items()}
        
        # Speaker summaries
        speaker_summaries = {}
        for speaker, emotions in speaker_emotions.items():
            speaker_avg = {emotion: np.mean(scores) if scores else 0.0 
                          for emotion, scores in emotions.items()}
            speaker_summaries[speaker] = {
                'average_emotions': speaker_avg,
                'dominant_emotion': max(speaker_avg, key=speaker_avg.get),
                'total_segments': len([r for r in fused_results if r['speaker'] == speaker])
            }
        
        return {
            'overall_emotion_distribution': emotion_averages,
            'dominant_conversation_emotion': max(emotion_averages, key=emotion_averages.get),
            'speaker_summaries': speaker_summaries,
            'total_speakers': len(speaker_summaries),
            'conversation_duration': max([r['end_time'] for r in fused_results]) if fused_results else 0
        }
