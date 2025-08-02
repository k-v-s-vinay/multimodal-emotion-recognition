# test_whisper.py
import whisper
import os

def test_whisper():
    print("Testing Whisper installation...")
    
    try:
        # Load Whisper model
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully!")
        
        # Test with existing audio file
        test_files = ["test.wav", "sample.mp3", "audio.wav"]
        
        for filename in test_files:
            if os.path.exists(filename):
                print(f"Testing with {filename}...")
                result = model.transcribe(filename)
                print(f"✅ Transcription: {result['text']}")
                return True
        
        print("❌ No test audio files found")
        print("Please place a small audio file (test.wav) in the backend folder")
        return False
        
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
        return False

if __name__ == "__main__":
    test_whisper()
