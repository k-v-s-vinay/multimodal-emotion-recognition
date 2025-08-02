# Test script to verify all core imports for multimodal emotion recognition
import sys
print("🧪 TESTING CORE IMPORTS FOR MULTIMODAL EMOTION RECOGNITION\n")

def test_import(module_name, alias=None):
    try:
        if alias:
            exec(f"import {module_name} as {alias}")
            print(f"✅ {module_name} (as {alias})")
        else:
            __import__(module_name)
            print(f"✅ {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {str(e)}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} - {str(e)}")
        return False

# Test core web framework
print("🌐 WEB FRAMEWORK:")
test_import('flask')
test_import('flask_cors')

print("\n🧮 SCIENTIFIC COMPUTING:")
test_import('numpy', 'np')
test_import('pandas', 'pd')
test_import('scipy')
test_import('sklearn')

print("\n🤖 DEEP LEARNING:")
test_import('torch')
test_import('torchaudio')
test_import('tensorflow', 'tf')
test_import('transformers')

print("\n🎵 AUDIO PROCESSING:")
test_import('librosa')
test_import('whisper')
test_import('pyannote.audio')

print("\n👁️ COMPUTER VISION:")
test_import('cv2')
test_import('PIL')
test_import('deepface')
test_import('face_recognition')
test_import('dlib')

print("\n📊 DATA & VISUALIZATION:")
test_import('matplotlib.pyplot', 'plt')

print("\n💾 DATABASE (Missing - needs installation):")
test_import('pymongo')
test_import('dotenv')

print("\n🔧 SYSTEM INFO:")
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")