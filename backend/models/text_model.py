from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class TextEmotionModel:
    def __init__(self, model_name="bhadresh-savani/bert-base-uncased-emotion"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.emotion_pipeline = pipeline("text-classification", 
                                       model=self.model, 
                                       tokenizer=self.tokenizer,
                                       return_all_scores=True)
        
        # Alternative model for more nuanced emotion detection
        self.enhanced_pipeline = pipeline("text-classification",
                                        model="ayoubkirouane/BERT-Emotions-Classifier",
                                        return_all_scores=True)
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        return text
    
    def predict_emotion_basic(self, text):
        """Basic emotion prediction using BERT"""
        if not text or text.strip() == "":
            return {"neutral": 1.0}
        
        try:
            results = self.emotion_pipeline(text)
            emotion_scores = {result['label']: result['score'] for result in results[0]}
            return emotion_scores
        except Exception as e:
            print(f"Error in basic emotion prediction: {e}")
            return {"neutral": 1.0}
    
    def predict_emotion_enhanced(self, text):
        """Enhanced emotion prediction with more categories"""
        if not text or text.strip() == "":
            return {"neutral": 1.0}
        
        try:
            results = self.enhanced_pipeline(text)
            emotion_scores = {result['label']: result['score'] for result in results[0]}
            return emotion_scores
        except Exception as e:
            print(f"Error in enhanced emotion prediction: {e}")
            return self.predict_emotion_basic(text)
    
    def analyze_sentiment_context(self, text):
        """Analyze sentiment and emotional context"""
        # Basic sentiment analysis
        sentiment_pipeline = pipeline("sentiment-analysis")
        sentiment = sentiment_pipeline(text)[0]
        
        # Extract emotional keywords
        emotional_keywords = {
            'positive': ['happy', 'joy', 'excited', 'love', 'amazing', 'wonderful', 'great'],
            'negative': ['sad', 'angry', 'hate', 'terrible', 'awful', 'horrible', 'bad'],
            'neutral': ['okay', 'fine', 'normal', 'average', 'usual']
        }
        
        text_lower = text.lower()
        keyword_matches = {}
        for emotion_type, keywords in emotional_keywords.items():
            matches = [word for word in keywords if word in text_lower]
            keyword_matches[emotion_type] = matches
        
        return {
            'sentiment': sentiment,
            'keyword_matches': keyword_matches,
            'text_length': len(text),
            'word_count': len(text.split())
        }
    
    def process_conversation_text(self, speaker_segments):
        """Process text from conversation segments"""
        results = []
        
        for segment in speaker_segments:
            text = segment.get('text', '')
            if not text:
                continue
            
            # Preprocess text
            clean_text = self.preprocess_text(text)
            
            # Get emotion predictions
            basic_emotions = self.predict_emotion_basic(clean_text)
            enhanced_emotions = self.predict_emotion_enhanced(clean_text)
            
            # Analyze context
            context_analysis = self.analyze_sentiment_context(text)
            
            # Combine results
            segment_result = {
                'speaker': segment.get('speaker'),
                'start_time': segment.get('start_time'),
                'end_time': segment.get('end_time'),
                'original_text': text,
                'cleaned_text': clean_text,
                'basic_emotions': basic_emotions,
                'enhanced_emotions': enhanced_emotions,
                'context_analysis': context_analysis,
                'dominant_text_emotion': max(basic_emotions, key=basic_emotions.get)
            }
            
            results.append(segment_result)
        
        return results
    
    def get_conversation_summary(self, processed_segments):
        """Generate overall conversation emotion summary"""
        if not processed_segments:
            return {}
        
        # Aggregate emotions across all segments
        all_emotions = {}
        speaker_emotions = {}
        
        for segment in processed_segments:
            speaker = segment['speaker']
            if speaker not in speaker_emotions:
                speaker_emotions[speaker] = []
            
            # Collect basic emotions
            for emotion, score in segment['basic_emotions'].items():
                if emotion not in all_emotions:
                    all_emotions[emotion] = []
                all_emotions[emotion].append(score)
                speaker_emotions[speaker].append((emotion, score))
        
        # Calculate averages
        emotion_averages = {emotion: np.mean(scores) for emotion, scores in all_emotions.items()}
        
        # Get dominant emotion per speaker
        speaker_dominant = {}
        for speaker, emotions in speaker_emotions.items():
            if emotions:
                speaker_emotion_avg = {}
                for emotion, score in emotions:
                    if emotion not in speaker_emotion_avg:
                        speaker_emotion_avg[emotion] = []
                    speaker_emotion_avg[emotion].append(score)
                
                speaker_avg = {emotion: np.mean(scores) for emotion, scores in speaker_emotion_avg.items()}
                speaker_dominant[speaker] = max(speaker_avg, key=speaker_avg.get)
        
        return {
            'overall_emotion_distribution': emotion_averages,
            'dominant_conversation_emotion': max(emotion_averages, key=emotion_averages.get),
            'speaker_dominant_emotions': speaker_dominant,
            'total_segments': len(processed_segments)
        }
