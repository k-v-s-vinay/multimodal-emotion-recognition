# 🎯 Multimodal Emotion Recognition for Deaf Community

> **AI-powered conversation analysis system that provides visual emotional context for deaf and hard-of-hearing individuals in group conversations.**

[![React](https://img.shields.io/badge/React-18.2.0-blue.svg)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)](https://flask.palletsprojects.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

## 📸 Screenshots

![Dashboard Overview](screenshots/dashboard-overview.png)
*Multimodal emotion analysis dashboard with visual timeline*

## 🌟 Features

- **🎤 Audio Processing**: Speaker diarization, transcription, and emotion recognition
- **🎬 Video Analysis**: Facial emotion detection and person tracking
- **🧠 Multimodal Fusion**: Combines audio and video emotions for comprehensive analysis
- **📊 Visual Dashboard**: Accessible interface with emojis, timelines, and color coding
- **📄 Export Functionality**: Download transcripts with emotional context
- **♿ Accessibility Focused**: Designed specifically for deaf community needs

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **OpenAI Whisper** - Speech recognition
- **Pyannote.audio** - Speaker diarization
- **DeepFace** - Facial emotion recognition
- **Librosa** - Audio feature extraction
- **PyTorch** - Deep learning framework
- **MongoDB** - Database storage

### Frontend
- **React 18.2+**
- **TailwindCSS** - Styling
- **Axios** - HTTP requests
- **JavaScript ES6+**

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- FFmpeg (for audio/video processing)

### 1. Clone Repository
git clone https://github.com/k-v-s-vinay/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition


### 2. Backend Setup
cd backend
python -m venv venv

Windows
venv\Scripts\activate

macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
python app.py

### 3. Frontend Setup
cd frontend
npm install
npm start

### 4. Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## 📊 Usage

1. **Upload Media**: Upload audio or video files through the web interface
2. **View Analysis**: Explore results through interactive dashboard tabs:
   - **Overview**: Summary statistics and emotion distribution
   - **Timeline**: Visual conversation flow with timestamps
   - **Transcript**: Full transcript with speaker identification
   - **Insights**: Conversation analytics and patterns
3. **Export Results**: Download transcripts and analysis data

## 🏗️ Project Structure

multimodal-emotion-recognition/
├── backend/ # Flask API server
│ ├── models/ # AI model implementations
│ ├── utils/ # Utility functions
│ └── app.py # Main Flask application
├── frontend/ # React web interface
│ ├── src/
│ │ ├── components/ # React components
│ │ └── App.js # Main React app
└── README.md

<!------------------------------------------------------------------------------ ----->
Setup Instructions
Backend Setup:
Create virtual environment:

bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Set up environment variables:

bash
export MONGODB_URI="mongodb://localhost:27017/emotion_recognition"
export SECRET_KEY="your-secret-key-here"
Run the Flask app:

bash
python app.py
Frontend Setup:
Install Node.js dependencies:

bash
cd frontend
npm install
Install and configure Tailwind CSS:

bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
Start the React app:

bash
npm start
VS Code Extensions Recommended:
Python

Pylance

ES7+ React/Redux/React-Native snippets

Tailwind CSS IntelliSense

MongoDB for VS Code

REST Client

This complete implementation provides a fully functional multimodal emotion recognition system with audio processing using Whisper and speaker diarization, text emotion analysis using BERT, facial emotion recognition using DeepFace, and multimodal fusion techniques. The system is designed to be accessible for deaf individuals with visual dashboards and emotion timelines.
