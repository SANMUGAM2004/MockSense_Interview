

# import json
# import os
# import google.generativeai as genai
# from django.http import JsonResponse
# from django.shortcuts import render
# from django.views.decorators.csrf import csrf_exempt
# from dotenv import load_dotenv

# # Load API Key
# load_dotenv()
# # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # GEMINI_API_KEY = "AIzaSyCUUmAWPdNRByj4hAtHntFZbZizGXuojVc"    // Please Uncomment this line for running grp discussion.
# GEMINI_API_KEY = "asd"
# genai.configure(api_key=GEMINI_API_KEY)

# # Function to generate AI responses
# def generate_ai_response(topic, previous_responses, role):
#     """Generate AI response for the discussion"""
#     model = genai.GenerativeModel("gemini-pro")

#     prompt = (
#         f"You are in a group discussion on '{topic}' as a {role}. "
#         "Engage in a natural, casual conversation. Keep it short, friendly, and relevant. "
#         "Previous responses:\n\n"
#         + "\n".join(previous_responses) +
#         "\nNow respond as a participant."
#     )

#     try:
#         response = model.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         return f"Error generating response: {str(e)}"

# # View to handle discussion page
# def discussion_page(request):
#     return render(request, "discussion.html")

# # API to handle user response
# @csrf_exempt  # Allow POST requests without CSRF for testing (use middleware in production)
# def group_discussion(request):
#     if request.method == "POST":
#         try:
#             data = json.loads(request.body)
#             topic = data.get("topic", "General")
#             user_response = data.get("user_response", "")

#             previous_responses = [f"You (User): {user_response}"]

#             # Generate AI responses
#             ai_male = generate_ai_response(topic, previous_responses, "Male Participant")
#             ai_female = generate_ai_response(topic, previous_responses, "Female Participant")

#             return JsonResponse({"ai_male": ai_male, "ai_female": ai_female})

#         except json.JSONDecodeError:
#             return JsonResponse({"error": "Invalid JSON"}, status=400)

#     return JsonResponse({"error": "Invalid request"}, status=400)

# def group_discussion_api(request):
#     if request.method == "POST":
#         return JsonResponse({"message": "API received data!"})
#     return JsonResponse({"error": "Invalid request"}, status=400)
# ===========================================================================================================
# import json
# import time
# import random
# import numpy as np
# import cv2
# import base64
# from django.http import JsonResponse, StreamingHttpResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.shortcuts import render
# from django.conf import settings
# from keras.models import model_from_json

# # Load Emotion Detection Model
# emotion_model_path = settings.BASE_DIR / "mock_interview/ML_model/emotiondetector_updated.json"
# emotion_weights_path = settings.BASE_DIR / "mock_interview/ML_model/emotiondetector_updated.h5"

# with open(emotion_model_path, "r") as json_file:
#     emotion_model = model_from_json(json_file.read())

# emotion_model.load_weights(emotion_weights_path)

# # Load Motivational Quotes
# with open(settings.BASE_DIR / "mock_interview/ML_model/motivational_quotes.json", "r") as file:
#     quotes_data = json.load(file)
# quotes = quotes_data["quotes"]

# # Define Emotion Labels
# labels = {0: "fear", 1: "confused", 2: "shy", 3: "neutral", 4: "happy"}

# # Load Haar Cascade for Face Detection
# haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# face_cascade = cv2.CascadeClassifier(haar_file)

# # Configuration
# CONFUSED_THRESHOLD = 5  # Confused for 5 seconds before showing a quote
# QUOTE_DISPLAY_DURATION = 7  # Quote stays for 7 seconds
# NEXT_QUOTE_DELAY = 5  # Cooldown before detecting confusion again

# # Variables for tracking quotes
# confused_start_time = None
# quote_display_time = None
# current_quote = None

# @csrf_exempt
# def detect_emotion(request):
#     """Detects emotion from an image sent via POST request."""
#     global confused_start_time, quote_display_time, current_quote
#     if request.method == "POST":
#         try:
#             data = json.loads(request.body)
            
#             if "image" not in data or not data["image"]:
#                 return JsonResponse({"error": "Missing 'image' data"}, status=400)

#             # Decode the Base64 image
#             image_data = base64.b64decode(data["image"].split(",")[1])
#             nparr = np.frombuffer(image_data, np.uint8)
#             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#             if img is None:
#                 return JsonResponse({"error": "Invalid image data"}, status=400)

#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#             detected_emotion = "neutral"
#             current_time = time.time()

#             for (x, y, w, h) in faces:
#                 face = gray[y:y + h, x:x + w]
#                 face = cv2.resize(face, (48, 48))
#                 face = np.expand_dims(face, axis=0)
#                 face = np.expand_dims(face, axis=-1) / 255.0  # Normalize

#                 emotion_prediction = emotion_model.predict(face)
#                 emotion_index = np.argmax(emotion_prediction)
#                 detected_emotion = labels[emotion_index]

#                 # Handle Motivational Quotes
#                 if quote_display_time is not None:
#                     if current_time - quote_display_time < QUOTE_DISPLAY_DURATION:
#                         return JsonResponse({"emotion": detected_emotion, "quote": current_quote})
#                     else:
#                         quote_display_time = None
#                         confused_start_time = None  # Reset confusion timer

#                 elif detected_emotion == "confused":
#                     if confused_start_time is None:
#                         confused_start_time = current_time
#                     elif current_time - confused_start_time >= CONFUSED_THRESHOLD:
#                         current_quote = random.choice(quotes)
#                         quote_display_time = current_time
#                         confused_start_time = None  # Reset timer
#                         return JsonResponse({"emotion": detected_emotion, "quote": current_quote})

#             return JsonResponse({"emotion": detected_emotion})

#         except json.JSONDecodeError:
#             return JsonResponse({"error": "Invalid JSON format"}, status=400)

#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=500)

#     return JsonResponse({"error": "Invalid request method"}, status=405)


# def discussion_view(request):
#     """Render the Group Discussion Page."""
#     return render(request, "group_discussion/discussion.html")

import json
import time
import random
import numpy as np
import cv2
import base64
import os
import google.generativeai as genai
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.conf import settings
from keras.models import model_from_json
from dotenv import load_dotenv
from gtts import gTTS
import logging

logger = logging.getLogger(__name__)

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
CONFUSED_THRESHOLD = 5  # Confused for 5 seconds before showing a quote
QUOTE_DISPLAY_DURATION = 7  # Quote stays for 7 seconds
NEXT_QUOTE_DELAY = 5  # Cooldown before detecting confusion again

# Variables for tracking quotes
confused_start_time = None
quote_display_time = None
current_quote = None

@csrf_exempt
def detect_emotion(request):
    """Detects emotion from an image sent via POST request."""
    global confused_start_time, quote_display_time, current_quote
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            
            if "image" not in data or not data["image"]:
                return JsonResponse({"error": "Missing 'image' data"}, status=400)

            # Decode the Base64 image
            image_data = base64.b64decode(data["image"].split(",")[1])
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return JsonResponse({"error": "Invalid image data"}, status=400)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            detected_emotion = "neutral"
            current_time = time.time()

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=-1) / 255.0  # Normalize

                emotion_prediction = emotion_model.predict(face)
                emotion_index = np.argmax(emotion_prediction)
                detected_emotion = labels[emotion_index]

                # Handle Motivational Quotes
                if quote_display_time is not None:
                    if current_time - quote_display_time < QUOTE_DISPLAY_DURATION:
                        return JsonResponse({"emotion": detected_emotion, "quote": current_quote})
                    else:
                        quote_display_time = None
                        confused_start_time = None  # Reset confusion timer

                elif detected_emotion == "confused":
                    if confused_start_time is None:
                        confused_start_time = current_time
                    elif current_time - confused_start_time >= CONFUSED_THRESHOLD:
                        current_quote = random.choice(quotes)
                        quote_display_time = current_time
                        confused_start_time = None  # Reset timer
                        return JsonResponse({"emotion": detected_emotion, "quote": current_quote})

            return JsonResponse({"emotion": detected_emotion})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)


def discussion_view(request):
    """Render the Group Discussion Page."""
    return render(request, "group_discussion/discussion.html")

# Load API Key
load_dotenv()
# GEMINI_API_KEY = "AIzaSyCUUmAWPdNRByj4hAtHntFZbZizGXuojVc"
GEMINI_API_KEY = "asdf"

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "asd")  # Default key for testing
genai.configure(api_key=GEMINI_API_KEY)

# Function to generate AI responses
def generate_ai_response(topic, previous_responses, role):
    """Generate AI response for the discussion"""
    model = genai.GenerativeModel("gemini-pro")

    prompt = (
        f"You are in a group discussion on '{topic}' as a {role}. "
        "Engage in a natural, casual conversation. Keep it short, friendly, and relevant. "
        "Previous responses:\n\n"
        + "\n".join(previous_responses) +
        "\nNow respond as a participant."
    )

    try:
        response = model.generate_content(prompt)
        print(response)
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"

# API to handle user response
@csrf_exempt  # Allow POST requests without CSRF for testing (use middleware in production)
def group_discussion(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            topic = data.get("topic", "General")
            user_response = data.get("user_response", "")

            previous_responses = [f"You (User): {user_response}"]
            logger.debug(f"Previous responses: {previous_responses}")  # Use logging
            print(f"Previous responses: {previous_responses}")  # Use print()

            # ai_male = generate_ai_response(topic, previous_responses, "Male Participant")
            # ai_female = generate_ai_response(topic, previous_responses, "Female Participant")
            ai_male = "Hello"
            print(f"AI Male Response: {ai_male}")  # ✅ DEBUG
            print(f"AI Female Response: {ai_male}")  # ✅ DEBUg
            
            return JsonResponse({"ai_male": ai_male, "ai_female": ai_female})

        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
            return JsonResponse({"error": "Invalid JSON"}, status=400)

    return JsonResponse({"error": "Invalid request"}, status=400)

def group_discussion_api(request):
    if request.method == "POST":
        return JsonResponse({"message": "API received data!"})
    return JsonResponse({"error": "Invalid request"}, status=400)
