# My Lia Project

## Overview

PodManagerAI Transcription Tool is an AI-driven editing platform that combines transcription, audio enhancement, AI-powered cutting, and analysis to optimize podcast production. It uses Flask as the backend, MongoDB for data storage, and a Streamlit-based frontend for an interactive user experience.

A complete AI-driven podcast editing platform covering the entire workflow: Transcription → AI Analysis → Editing → Export. With both automatic AI enhancements and manual adjustments, users gain maximum control over their audio and video productions.


## 6 kraftfulla AI-funktioner

PodManager AI Transcription Tool consists of six key interactive features where users can process audio and video. A seventh feature, AI Video Cutting, is on the way!

1️⃣ 🎙 AI-Powered Transcription

Transcription of audio and video
Speaker identification & timestamps
Translation & AI-enhanced text improvements
2️⃣ 🎛 Audio Enhancement & AI Analysis

Noise reduction & audio enhancement
Sentiment analysis & AI-powered audio quality assessment
Speech rate analysis & AI-driven recommendations
3️⃣ ✂ Audio Cutting

Manual audio trimming with waveform visualization
Adjust clip times using sliders
4️⃣ 🤖 AI Audio Cutting

Automatic detection of long pauses & filler words
AI-suggested optimal cuts
5️⃣ 📹 Video Enhancement & AI Analysis

AI-powered video quality enhancement
AI analysis of content (emotion detection, sentiment analysis)
6️⃣ 📼 Video Cutting

Manual video editing and trimming tools
Time-based cutting & export
🔜 7️⃣ AI Video Cutting (Coming Soon!)

AI-driven automated video cutting
AI-based detection of irrelevant segments


## Core Features

✅ AI-powered transcription – Converts audio and video to text with speaker recognition
✅ AI-driven audio enhancement – Removes noise, improves clarity, and balances audio levels
✅ AI-powered audio & video cutting – Automated and manual editing options
✅ Automatic detection of filler words & long pauses – Removes unnecessary parts from audio
✅ AI-generated show notes & marketing snippets – Creates summaries and highlights
✅ Export edited files in multiple formats – Download optimized versions




## Features

🎙 AI Transcription & Text Processing

Speaker diarization: Identifies different speakers in a recording
Precise timestamps for every word and sentence
Support for multiple languages via Whisper & OpenAI
AI-powered removal of filler words and incoherent sentences
AI-generated transcription improvements

🎛 Audio & Video Enhancement

Noise reduction (NSNet2, Noisereduce, FFmpeg)
AI-driven volume adjustment and normalization
Background noise analysis (classification of noise disturbances)
Audio quality improvement using SoX & Librosa

✂ Audio & Video Cutting

Automatic detection of long pauses & filler words
AI-suggested cutting based on speech analysis
Manual adjustments with waveform UI & sliders
Background analysis for clip recommendations

📊 AI Analysis & Quality Assessment

Speech sentiment analysis (positive, neutral, negative)
Speech rate analysis: WPM (Words Per Minute) calculation
AI-based assessment of audio quality & clarity score
Zero-shot classification to evaluate sentence relevance

📝 AI-Generated Content & Export

Show notes with summaries & highlights
Automatically generated social media snippets
Ability to translate transcriptions into multiple languages
Export transcriptions in TXT, SRT, & JSON formats
Audio & video files can be downloaded in optimized formats
🖥 Interactive UI & User Experience

Streamlit-based frontend with simple buttons & sliders
Live preview of AI transcriptions and suggestions
Ability to preview & adjust cuts

📡 Backend & Infrastructure

Flask API for handling transcription, analysis, and processing
MongoDB (GridFS) for storing and managing files
Azure Speech-to-Text & OpenAI Whisper as AI engines
FFmpeg & SoX for audio & video editing
Logging & debugging using Python’s logging module


## Technologies Used

Backend & Database Management
Python (Flask) – API handling and backend logic
MongoDB (GridFS) – Database for storing audio and video files
FFmpeg – Audio & video processing (noise reduction, cutting, audio enhancement)
Pydub – Audio analysis and manipulation
MoviePy – Video editing
SoX & Noisereduce – Advanced noise reduction
Librosa & Torchaudio – Audio analysis and feature extraction


AI & NLP
ElevenBase & Whisper – AI-powered transcription and speaker recognition
OpenAI GPT-4 – AI-generated suggestions, show notes & content generation
Hugging Face Transformers (BERT/RoBERTa, MiniLM) – NLP-based analysis
Pyannote – Speaker recognition and diarization
Azure Speech-to-Text – Alternative transcription solution
TextBlob & TextStat – Sentiment analysis & readability assessment
DistilBERT & Zero-shot Classification – AI-based relevance analysis
Emotion Detection Models – Emotion analysis in text and speech


Frontend & UI
Streamlit – Interactive frontend
HTML, CSS, JavaScript – Custom UI components
Jinja Templates – Dynamic UI rendering


Other Tools & Infrastructure
Miniconda – Virtual environment management
Azure DevOps & GitHub – Version control and CI/CD
GridFS (MongoDB) – Handling of large audio & video files
Matplotlib & NumPy – Visual analysis and waveform visualization

# Getting Started

first, create a .env file in root.
second, add all your keys in the .env file, you have examples in .env.example
you need to add:

# Database connection uri
MONGODB_URI="YOUR_MONGODB_URI_HERE"

# Local or Production Base URLs
LOCAL_BASE_URL="http://127.0.0.1:8000"

# Api Keys
HF_TOKEN="YOUR HF TOKEN" #HUGGING FACE

OPENAI_API_KEY="YOUR OPENAI API KEY"

ELEVENLABS_API_KEY="YOUR ELEVENBASE TOKEN"

1. open terminal type:
conda env create -f environment.yml

(This could take sometime cause its quite big.)
(This will set up the miniconda enviroment)


2. type:
conda activate aitranscription

(now your miniconda enviorment is set and running).


4. type:
python -r requriments.txt

(to install all the packages)


5. type:
python app.py

(to run the app)

6. open another terminal and type:
streamlit run streamlit_transcription.py

(now navigate to the streamlit page you can navigate to it from terminal or a page should be open automaticaly)

# Contact

pete.molen@gmail.com

