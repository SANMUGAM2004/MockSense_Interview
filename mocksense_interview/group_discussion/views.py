

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
from django.conf import settings
BASE_DIR = settings.BASE_DIR

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
    return render(request, "group_discussion/index.html")

# def grp_discussion_view(request):
#     """Render the Group Discussion Page."""
#     return render(request, "group_discussion/discussion.html")
GEMINI_API_KEY = "AIzaSyCUUmAWPdNRByj4hAtHntFZbZizGXuojVc"
# GEMINI_API_KEY = "acac"
# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-pro")

def start_grp_discussion_view(request):
    """Render the Group Discussion Page with generated topic and discussion."""

    # prompt = (
    #     "Generate a random group discussion topic. It can be either a social or technological topic.\n\n"
    #     "Then, simulate a group discussion between two participants: Male and Female.\n"
    #     "Each person should respond with 5 professional points.\n"
    #     "Some points can be in agreement, others can respectfully oppose.\n"
    #     "The tone should be formal, insightful, and natural."
    #     "\n\nProvide your output in the following format:\n"
    #     "Group Discussion Topic: <Topic>\n\n"
    #     "Male:\n1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n\n"
    #     "Female:\n1. ...\n2. ...\n3. ...\n4. ...\n5. ..."
    # )

    # response = model.generate_content(prompt)

    # # Parse the result
    # lines = response.text.strip().splitlines()
    topic = ""
    male_points = []
    female_points = []
    current_speaker = None

    # for line in lines:
    #     line = line.strip()
    #     if line.lower().startswith("group discussion topic"):
    #         topic = line.split(":", 1)[1].strip()
    #     elif line.lower().startswith("male"):
    #         current_speaker = "male"
    #     elif line.lower().startswith("female"):
    #         current_speaker = "female"
    #     elif line and line[0].isdigit() and "." in line:
    #         point = line.split(".", 1)[1].strip()
    #         if current_speaker == "male":
    #             male_points.append(point)
    #         elif current_speaker == "female":
    #             female_points.append(point)

    # Write the result to response.txt (overwrite if exists)
    # with open("response.txt", "w", encoding="utf-8") as file:
    #     file.write(f"Group Discussion Topic: {topic}\n\n")
    #     file.write("Male:\n")
    #     for idx, point in enumerate(male_points, 1):
    #         file.write(f"{idx}. {point}\n")
    #     file.write("\nFemale:\n")
    #     for idx, point in enumerate(female_points, 1):
    #         file.write(f"{idx}. {point}\n")

    # return render(request, "group_discussion/discussion.html", {
    #     "topic": topic,
    #     "male_points": male_points,
    #     "female_points": female_points,
    # })
    topic = "The Impact of Artificial Intelligence on the Future of Work"
    print("=========================",topic,"=============================")
    return render(request, "group_discussion/brainstorming.html", {"topic": topic})

# def group_discussion_view(request):
#     return render(request, "group_discussion/discussion.html")
def group_discussion_view(request):
    topic = ""
    male_points = []
    female_points = []
    
    # Make sure this path points to where your response.txt is stored
    file_path = os.path.join(BASE_DIR, "group_discussion", "response.txt")


    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Group Discussion Topic:"):
                topic = line.replace("Group Discussion Topic:", "").strip()
            elif line.startswith("Male:"):
                current_section = "male"
            elif line.startswith("Female:"):
                current_section = "female"
            elif current_section == "male" and line[0].isdigit():
                male_points.append(line[line.find('.')+1:].strip())
            elif current_section == "female" and line[0].isdigit():
                female_points.append(line[line.find('.')+1:].strip())

    context = {
        "topic": topic,
        "male_points": male_points,
        "female_points": female_points
    }

    return render(request, "group_discussion/discussion.html", context)

@csrf_exempt
# def save_user_gd_response(request):
#     if request.method == "POST":
#         data = json.loads(request.body)
#         topic = data.get("topic", "")
#         responses = data.get("user_responses", [])
#         # settings.BASE_DIR / "mock_interview/ML_model/emotiondetector_updated.json
#         response_path = os.path.join(settings.BASE_DIR / "group_discussion", "user_gd_response.txt")
#         with open(response_path, "w", encoding="utf-8") as file:
#             file.write(f"Topic: {topic}\n\n")
#             for item in responses:
#                 file.write(f"Round {item['round']}: {item['response']}\n")

#         return JsonResponse({"status": "success", "message": "Responses saved."})
#     return JsonResponse({"status": "error", "message": "Invalid request."}, status=400)
def save_user_gd_response(request):
    if request.method == "POST":
        data = json.loads(request.body)
        topic = data.get("topic", "")
        responses = data.get("user_responses", [])

        response_path = os.path.join(settings.BASE_DIR / "group_discussion", "user_gd_response.txt")
        with open(response_path, "w", encoding="utf-8") as file:
            file.write(f"Topic: {topic}\n\n")
            for item in responses:
                file.write(f"Round {item['round']}: {item['response']}\n")
        print("00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")

        # 🌟 Improve sentences using Gemini
        try:
            with open(response_path, "r", encoding="utf-8") as file:
                user_text = file.read()
                
            print("user test is finished===============================================================")

            prompt = f"""Improve the clarity, grammar, and structure of these responses for a group discussion.
            Return the improved sentences as a bullet list. Don't change the context.

            {user_text}
            """
            response = model.generate_content(prompt)
            improved_sentences = response.text.strip()
            print("=+++++++++++++++++++++++++++++++++++++++++++++++",response)

            # Save improved output
            improved_path = os.path.join(settings.BASE_DIR / "group_discussion", "improved_response.txt")
            with open(improved_path, "w", encoding="utf-8") as file:
                file.write(improved_sentences)

        except Exception as e:
            print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            return JsonResponse({"status": "error", "message": str(e)}, status=500)

        return JsonResponse({"status": "success", "message": "Responses saved and improved."})

    return JsonResponse({"status": "error", "message": "Invalid request."}, status=400)

def improve_sentences_view(request):
    user_path = os.path.join(settings.BASE_DIR / "group_discussion", "user_gd_response.txt")
    improved_path = os.path.join(settings.BASE_DIR / "group_discussion", "improved_response.txt")

    with open(user_path, "r", encoding="utf-8") as file:
        user_text = file.read()

    with open(improved_path, "r", encoding="utf-8") as file:
        improved_sentences = file.read()

    return render(request, "group_discussion/result.html", {
        "original": user_text,
        "improved": improved_sentences
    })


# ===========================================================
# Load API Key
load_dotenv()
# GEMINI_API_KEY = "AIzaSyCUUmAWPdNRByj4hAtHntFZbZizGXuojVc"
# GEMINI_API_KEY = "asdf"

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
