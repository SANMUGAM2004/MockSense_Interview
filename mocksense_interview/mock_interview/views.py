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
import sklearn
import spacy
from django.core.files.storage import default_storage
from pdfminer.high_level import extract_text
from pathlib import Path
import json
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load pre-trained model and tokenizer
question_generator_path = settings.BASE_DIR / "mock_interview/ML_model/question_generator_rnn.h5"
model = tf.keras.models.load_model(question_generator_path)
with open(settings.BASE_DIR / "mock_interview/ML_model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open(settings.BASE_DIR / "mock_interview/ML_model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
    
# Load intents dataset
with open(settings.BASE_DIR / "mock_interview/ML_model/data/intents.json", 'r') as file:
    data = json.load(file)

# Extract max_length from the model's training phase
max_length = 20  # Make sure this matches the one used in training

# **Function to Generate a Question**
def generate_question(user_response):
    # Tokenize and pad user response
    input_sequence = tokenizer.texts_to_sequences([user_response])
    input_padded = pad_sequences(input_sequence, maxlen=max_length, padding='post')

    # Predict topic (tag) using the model
    predicted_index = np.argmax(model.predict(input_padded))
    predicted_tag = label_encoder.inverse_transform([predicted_index])[0]

    # Mapping questions to responses
    question_response_map = {}
    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            for pattern in intent['patterns']:
                question_response_map[pattern] = intent['responses'][0]  # Assuming one response per tag

    # Select a random question
    if not question_response_map:
        return "Sorry, I couldn't determine the topic from your response.", None

    question = random.choice(list(question_response_map.keys()))
    answer = question_response_map[question]
    return question, answer

# **Function to Generate 5 Questions**
def generate_questions_from_resume(resume_json):
    with open(resume_json, 'r') as file:
        resume_data = json.load(file)

    skills = resume_data.get("Skills", [])
    projects = resume_data.get("Projects", [])

    # Combine both skills and projects to generate questions
    topics = skills + projects
    if len(topics) < 5:
        topics = topics * (5 // len(topics) + 1)  # Duplicate if less than 5

    topics = topics[:5]  # Ensure only 5 topics are used

    questions_answers = []
    for topic in topics:
        question, answer = generate_question(topic)
        questions_answers.append({"question": question, "answer": answer})

    return questions_answers

# # **Usage Example**
# resume_file = "resume_data.json"
# questions_with_answers = generate_questions_from_resume(resume_file)

# **Print Questions & Answers**
# for idx, qa in enumerate(questions_with_answers, 1):
#     print(f"Q{idx}: {qa['question']}")
#     print(f"A{idx}: {qa['answer']}\n")
# ================================================================================================

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

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
with open(settings.BASE_DIR / "mock_interview/ML_model/data/keywords.json", "r") as file:
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

# View: Render Quiz Page
def quiz_view(request):
    return render(request, "mock_interview/quiz.html", {'proctored': True})

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
    return render(request, "mock_interview/quiz.html", {'proctored': True})

def index(request):
    return render(request,"mock_interview/index.html", {"proctored": True} )

def resume_uploading(request):
    return render(request,"mock_interview/resume_upload.html")


@csrf_exempt
def upload_resume(request):
    if request.method == "POST" and request.FILES.get("resume"):
        uploaded_file = request.FILES["resume"]
        file_path = default_storage.save("resumes/" + uploaded_file.name, uploaded_file)
        print("Resume come for extraction....................")    #Comments

        # Extract text from PDF
        resume_text = extract_text(Path(settings.MEDIA_ROOT) / file_path)

        # Extract details from resume
        parsed_data = extract_resume_entities(resume_text)
        
        # ✅ Save extracted data to JSON
        json_path = Path(settings.MEDIA_ROOT) / "resume_data.json"
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(parsed_data, json_file, indent=4)
            
        resume_file = "resume_data.json"
        questions_with_answers = generate_questions_from_resume(resume_file)
        # Save generated questions to JSON file
        save_questions_and_answers(questions_with_answers)
        # ###################Printing the generated questions.
        # for idx, qa in enumerate(questions_with_answers, 1):
        #     print(f"Q{idx}: {qa['question']}")
        #     print(f"A{idx}: {qa['answer']}\n")
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


def save_questions_and_answers(questions_with_answers):
    """
    Saves questions to 'question.json' and answers to 'answers.json'.
    """
    question_path = settings.BASE_DIR / "mock_interview/ML_model/questions.json"
    answer_path = settings.BASE_DIR / "mock_interview/ML_model/answers.json"

    # Extract questions and answers
    questions = [qa["question"] for qa in questions_with_answers]
    answers = [qa["answer"] for qa in questions_with_answers]

    # Save questions to question.json
    with open(question_path, "w", encoding="utf-8") as q_file:
        json.dump({"questions": questions}, q_file, indent=4)

    # Save answers to answers.json
    with open(answer_path, "w", encoding="utf-8") as a_file:
        json.dump({"answers": answers}, a_file, indent=4)

    print(f"✅ Questions saved to {question_path}")
    print(f"✅ Answers saved to {answer_path}")
    
    #===================================================================


# Set up logging
logger = logging.getLogger(__name__)

# Function to load correct answers
def load_correct_answers(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)  # Load JSON file
            if isinstance(data, dict) and "answers" in data:
                return data["answers"]  # Extract and return the answers list
            else:
                logger.error("❌ Invalid format: Expected a dictionary with 'answers' key")
                return []
    except FileNotFoundError:
        logger.error(f"❌ File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON decoding error in {file_path}: {e}")
        return []

# Function to load user answers
def load_user_answers(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)  # Load entire JSON array
            if isinstance(data, list):
                return [entry["answer"] for entry in data if "answer" in entry]  # Extract user answers
            else:
                logger.error("❌ Invalid format: Expected a list of dictionaries")
                return []
    except FileNotFoundError:
        logger.error(f"❌ File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON decoding error in {file_path}: {e}")
        return []

# Function to calculate similarity using spaCy
def calculate_similarity(answer, user_answer):
    doc1 = nlp(answer)
    doc2 = nlp(user_answer)
    return doc1.similarity(doc2)

def limit_exceeded(request):
    return render(request, "limit_exceeded.html", {"proctored": True})

# Function to process interview results
def mock_interview_result(request):
    # Read the content from the text file
    with open(settings.BASE_DIR / "mock_interview/user_answers.json", 'r') as file:
        lines = file.readlines()

    # Strip each line of leading/trailing whitespace
    json_array = [line.strip() for line in lines]

    # Wrap the entire result in square brackets and join without a trailing comma
    formatted_json = f"[{', '.join(json_array)}]"

    # Save the formatted content to a new file
    with open(settings.BASE_DIR / "mock_interview/user_answers.json", 'w') as output_file:
        output_file.write(formatted_json)

    # Load correct answers and user answers
    correct_answers = load_correct_answers(settings.BASE_DIR / "mock_interview/ML_model/answers.json")
    user_answers = load_user_answers(settings.BASE_DIR / "mock_interview/user_answers.json")

    # print("✅ Correct Answers:", correct_answers)
    # print("✅ User Answers:", user_answers)

    if not correct_answers or not user_answers:
        return render(request, "result.html", {"error": "Missing answer files or invalid data format!"})

    # Compute similarity scores
    similarity_scores = [calculate_similarity(ca, ua) for ca, ua in zip(correct_answers, user_answers)]
    overall_score = np.mean(similarity_scores) * 100  # Convert to percentage

    # Prepare data for chart
    result_data = {
        "similarity_scores": [round(score * 100, 2) for score in similarity_scores],  # Convert to percentage
        "overall_score": round(overall_score, 2)
    }
    print("📊 Result Data:", result_data)

    return render(request, "mock_interview/result.html", {"result_data": result_data})