# Updated code========
import json
import random
import time
import pyttsx3
import cv2
import os
import threading
import logging
import numpy as np
import speech_recognition as sr
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.conf import settings
from keras.models import model_from_json
import spacy
from django.core.files.storage import default_storage
from pdfminer.high_level import extract_text
from pathlib import Path

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load Emotion Detection Model
emotion_model_path = settings.BASE_DIR / "mock_interview/ML_model/emotiondetector_updated.json"
emotion_weights_path = settings.BASE_DIR / "mock_interview/ML_model/emotiondetector_updated.h5"

with open(emotion_model_path, "r") as json_file:
    emotion_model = model_from_json(json_file.read())

emotion_model.load_weights(emotion_weights_path)

# Load Motivational Quotes
with open(settings.BASE_DIR / "mock_interview/ML_model/motivational_quotes.json", "r") as file:
    quotes_data = json.load(file)
quotes = quotes_data["quotes"]

# Define Emotion Labels
labels = {0: "fear", 1: "confused", 2: "shy", 3: "neutral", 4: "happy"}

# Load Haar Cascade for Face Detection
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# Configuration
CONFUSED_THRESHOLD = 5  # 5 seconds of confusion before showing a quote
QUOTE_DISPLAY_DURATION = 7  # Quote stays for 7 seconds
NEXT_QUOTE_DELAY = 5  # Cooldown before detecting confusion again
ANSWER_TIME_LIMIT = 10  # 1 minute for answering each question (modifiable)

# Variables for quote tracking
confused_start_time = None
quote_display_time = None
current_quote = None

# Load keywords for skills & projects
with open(settings.BASE_DIR / "mock_interview/ML_model/keywords.json", "r") as file:
    keywords = json.load(file)

#  Load questions from JSON file
def load_questions():
    """Dynamically load questions from JSON each time it's called"""
    questions_path = os.path.join(settings.BASE_DIR, "mock_interview", "ML_model", "questions.json")
    with open(questions_path, "r", encoding="utf-8") as file:
        questions_data = json.load(file)
    return questions_data["questions"]

def start_interview(request):
    """Return questions when the interview starts"""
     # Define the path for the user answers file
    file_path = os.path.join(settings.BASE_DIR, "mock_interview", "user_answers.json")
    
    # Delete the file if it exists (reset the answers before starting the interview)
    if os.path.exists(file_path):
        os.remove(file_path)
    questions = load_questions()  # Load questions dynamically
    return JsonResponse({"status": "started", "questions": questions})

# Save User's Answer
@csrf_exempt
def save_answer(request):
    """Saves the user's answer"""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            question_index = data["question_index"]
            answer = data["answer"]

            # Save answer in a JSON file (or database)
            with open("mock_interview/user_answers.json", "a") as file:
                json.dump({"question_index": question_index, "answer": answer}, file)
                file.write("\n")

            return JsonResponse({"status": "saved"})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request"}, status=400)


def listen_answer(request):
    """Captures user's voice and converts it into text"""
    if request.method == "POST":
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            try:
                audio = recognizer.listen(source, timeout=5)  # Listen for 5 seconds
                user_answer = recognizer.recognize_google(audio)  # Convert speech to text
                print("User said:", user_answer)
                
                return JsonResponse({"status": "completed", "answer": user_answer})
            except sr.UnknownValueError:
                return JsonResponse({"status": "completed", "answer": "Could not understand"})
            except sr.RequestError:
                return JsonResponse({"status": "error", "message": "Speech recognition API unavailable"})
    
    return JsonResponse({"error": "Invalid request"}, status=400)

# Set up logging
logger = logging.getLogger(__name__)

def mock_interview_result(request):
    """Reads the stored answers from user_answers.json"""
    # Define the file path
    file_path = os.path.join(settings.BASE_DIR, "mock_interview", "user_answers.json")

    # Log the file path
    logger.info(f"Attempting to read answers from: {file_path}")

    # Ensure the file exists before reading it
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}. Creating a new one.")
        with open(file_path, "w") as file:
            json.dump([], file)  # Create an empty list if the file doesn't exist

    try:
        with open(file_path, "r") as file:
            answers = json.load(file)  # Read JSON data
        logger.info(f"Answers read successfully: {answers}")
        return JsonResponse({"answers": answers})
    except Exception as e:
        # Log the exception for debugging
        logger.error(f"Error reading file {file_path}: {e}")
        return JsonResponse({"error": str(e)}, status=500)


# View: Render Quiz Page
def quiz_view(request):
    return render(request, "mock_interview/quiz.html")

# Function: Video Stream Generator
def generate_frames():
    global confused_start_time, quote_display_time, current_quote 
    cap = cv2.VideoCapture(0)  # Open Webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        current_time = time.time()

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1) / 255.0  # Normalize

            emotion_prediction = emotion_model.predict(face)
            emotion_index = np.argmax(emotion_prediction)
            emotion_text = labels[emotion_index]

            # Draw Rectangle and Display Emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Handle Motivational Quotes
            if quote_display_time is not None:
                if current_time - quote_display_time < QUOTE_DISPLAY_DURATION:
                    cv2.putText(frame, current_quote, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    quote_display_time = None
                    confused_start_time = None  # Reset confusion timer

            elif emotion_text == "confused":
                if confused_start_time is None:
                    confused_start_time = current_time
                elif current_time - confused_start_time >= CONFUSED_THRESHOLD:
                    current_quote = random.choice(quotes)
                    print(f"Motivational Quote: {current_quote}")
                    cv2.putText(frame, current_quote, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    quote_display_time = current_time
                    confused_start_time = None  # Reset timer

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

# View: Start Camera Feed
def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type="multipart/x-mixed-replace; boundary=frame")

# View: Render Quiz Page
def quiz_view(request):
    return render(request, "mock_interview/quiz.html")

def index(request):
    return render(request,"mock_interview/index.html" )

def resume_uploading(request):
    return render(request,"mock_interview/resume_upload.html")


@csrf_exempt
def upload_resume(request):
    if request.method == "POST" and request.FILES.get("resume"):
        uploaded_file = request.FILES["resume"]
        file_path = default_storage.save("resumes/" + uploaded_file.name, uploaded_file)

        # Extract text from PDF
        resume_text = extract_text(Path(settings.MEDIA_ROOT) / file_path)

        # Extract details from resume
        parsed_data = extract_resume_entities(resume_text)
        
        # ✅ Save extracted data to JSON
        json_path = Path(settings.MEDIA_ROOT) / "resume_data.json"
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(parsed_data, json_file, indent=4)

        return JsonResponse(parsed_data, safe=False)
    
    return JsonResponse({"error": "No file uploaded"}, status=400)

def extract_resume_entities(text):
    doc = nlp(text)
    extracted_data = {"Name": "", "Skills": [], "Projects": []}

    skillset, projectset = set(), set()

    # Extract name
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not extracted_data["Name"]:
            extracted_data["Name"] = ent.text

    # Extract Skills & Projects
    for line in text.split("\n"):
        for skill in keywords["skills"]:
            if skill.lower() in line.lower():
                skillset.add(skill)

        for project in keywords["projects"]:
            if project in line.lower():
                projectset.add(line.strip())

    extracted_data["Skills"] = list(skillset)
    extracted_data["Projects"] = [proj.replace("\u2022", "").strip() for proj in projectset]

    return extracted_data