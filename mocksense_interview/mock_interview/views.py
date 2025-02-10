# Updated code========

import json
import random
import time
import pyttsx3
import cv2
import threading
import numpy as np
import speech_recognition as sr
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.conf import settings
from keras.models import model_from_json

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

# Load Questions from JSON
with open(settings.BASE_DIR / "mock_interview/ML_model/questions.json", "r") as file:
    questions_data = json.load(file)
questions = questions_data["questions"]


# Initialize pyttsx3 TTS engine
engine = pyttsx3.init()

# Configuration for speech
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# ANSWER_TIME_LIMIT = 60  # Time limit for answering each question (modifiable)
ANSWER_TIME_LIMIT = 10  # 10 seconds for now for demonstration

# Function to speak text aloud (Text-to-Speech)
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Function to listen to the user's answer and transcribe it
def listen_answer():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        print("Listening for your answer...")
        try:
            audio = recognizer.listen(source, timeout=ANSWER_TIME_LIMIT)
            user_answer = recognizer.recognize_google(audio)
            print(f"User Answer: {user_answer}")
            return user_answer
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."
        except sr.RequestError:
            return "Speech recognition service unavailable."
        except sr.WaitTimeoutError:
            return "No speech detected."

# # Function to ask questions and collect answers
# def ask_questions(request):
#     question_index = int(request.GET.get('question_index', 0))  # Default to 0 if not provided
#     user_answers = []

#     # If we've reached the end of the questions, return the results
#     if question_index >= len(questions):
#         with open(settings.BASE_DIR / "mock_interview/useranswer.txt", "w") as file:
#             file.writelines(user_answers)
#         return JsonResponse({"status": "completed", "answers": user_answers})

#     # Get the next question
#     question = questions[question_index]
#     print(f"Asking Question: {question}")

#     # Trigger TTS to read the question aloud in a separate thread
#     speech_thread = threading.Thread(target=speak_text, args=(question,))
#     speech_thread.start()

#     # Wait for the question to be spoken (you can adjust this depending on the question length)
#     time.sleep(1)  # Sleep for a brief moment to let the question be heard

#     # Listen for the user's answer within the time limit
#     user_answer = listen_answer()

#     # Save the user's answer
#     user_answers.append(user_answer)
#     print({
#         "status": "question_asked",
#         "question": question,
#         "next_question_index": question_index + 1,
#         "user_answer": user_answer
#     })

#     # Send next question as JSON response
#     return JsonResponse({
#         "status": "question_asked",
#         "question": question,
#         "next_question_index": question_index + 1,
#         "user_answer": user_answer
#     })

# Function to ask questions and collect answers
@csrf_exempt
def ask_questions(request):
    question_index = int(request.GET.get('question_index', 0))  # Default to 0 if not provided
    user_answers = []

    # If we've reached the end of the questions, return the results
    if question_index >= len(questions):
        with open(settings.BASE_DIR / "mock_interview/useranswer.txt", "w") as file:
            file.writelines(user_answers)
        return JsonResponse({"status": "completed", "answers": user_answers})

    # Get the next question
    question = questions[question_index]
    print(f"Asking Question: {question}")

    # Trigger TTS to read the question aloud in a separate thread
    speech_thread = threading.Thread(target=speak_text, args=(question,))
    speech_thread.start()

    # Wait for the question to be spoken (adjust sleep time if needed)
    time.sleep(1)  # Brief pause to let the question be heard

    # Listen for the user's answer within the time limit
    user_answer = listen_answer()

    # Save the user's answer
    user_answers.append(user_answer)
    print(f"Answer for Question {question_index + 1}: {user_answer}")

    # Prepare the response for the next question
    next_question_index = question_index + 1

    # Send next question as JSON response to update the frontend
    response_data = {
        "status": "question_asked",
        "question": question,
        "next_question_index": next_question_index,
        "user_answer": user_answer
    }

    # You can now call this in your frontend to update the page dynamically with JavaScript

    return JsonResponse(response_data)

# View: Start Interview
def start_interview(request):
    # Trigger the question asking process by sending the first question
    return ask_questions(request)

# View: Render Quiz Page
def quiz_view(request):
    return render(request, "mock_interview/quiz.html")

# View: Show Results
def mock_interview_result(request):
    # Fetch answers from the file
    with open(settings.BASE_DIR / "mock_interview/useranswer.txt", "r") as file:
        user_answers = file.readlines()

    return render(request, "mock_interview/result.html", {"answers": user_answers})

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

# # Function: Listen and Transcribe Audio for Answers
# @csrf_exempt
# def listen_answer(request):
#     if request.method == "POST":
#         recognizer = sr.Recognizer()
#         with sr.Microphone() as source:
#             print("Listening for answer...")
#             recognizer.adjust_for_ambient_noise(source)

#             try:
#                 audio = recognizer.listen(source, timeout=5)  # Listen for 5 seconds
#                 user_answer = recognizer.recognize_google(audio)

#                 # Save to file
#                 with open(settings.BASE_DIR / "mock_interview/useranswer.txt", "a") as file:
#                     file.write(user_answer + "\n")

#                 return JsonResponse({"status": "completed", "answer": user_answer})

#             except sr.UnknownValueError:
#                 return JsonResponse({"status": "error", "message": "Could not understand audio"})
#             except sr.RequestError:
#                 return JsonResponse({"status": "error", "message": "Speech recognition service unavailable"})
#             except sr.WaitTimeoutError:
#                 return JsonResponse({"status": "error", "message": "No speech detected."})

# # Function to ask questions and collect answers
# def ask_questions(request):
#     question_index = 0
#     start_time = time.time()
#     user_answers = []

#     while question_index < len(questions) and time.time() - start_time < ANSWER_TIME_LIMIT:
#         question = questions[question_index]
#         print(f"Asking Question: {question}")

#         # Render question page with current question
#         return render(request, "mock_interview/question.html", {"question": question, "question_index": question_index + 1})

#         # After asking question, listen for the answer
#         response = listen_answer(request)
#         if response.status_code == 200:
#             user_answers.append(response.json()['answer'])
#         question_index += 1

#     # Save answers to file
#     with open(settings.BASE_DIR / "mock_interview/useranswer.txt", "w") as file:
#         file.writelines(user_answers)

#     return JsonResponse({"status": "completed", "answers": user_answers})

# View: Render Quiz Page
# def quiz_view(request):
#     return render(request, "mock_interview/quiz.html")

# View: Show Results
# def mock_interview_result(request):
#     # Fetch answers from the file
#     with open(settings.BASE_DIR / "mock_interview/useranswer.txt", "r") as file:
#         user_answers = file.readlines()

#     return render(request, "mock_interview/result.html", {"answers": user_answers})

# # In views.py
# def start_interview(request):
#     # Trigger the question asking process
#     return ask_questions(request)
