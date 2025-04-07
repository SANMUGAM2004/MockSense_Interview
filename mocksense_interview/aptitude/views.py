import json
import random
from django.shortcuts import render, redirect
from django.http import JsonResponse
import os

# Load questions from JSON
def load_questions():
    try:
        # Get the absolute path to the JSON file
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Get current app directory
        file_path = os.path.join(base_dir, "data", "questions.json")  # Construct full path
        
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

# Get random questions based on difficulty levels
def get_random_questions(data, category):
    easy_qns = random.sample(data[category]["Easy"], min(3, len(data[category]["Easy"])))
    medium_qns = random.sample(data[category]["Medium"], min(4, len(data[category]["Medium"])))
    hard_qns = random.sample(data[category]["Hard"], min(3, len(data[category]["Hard"])))

    all_questions = easy_qns + medium_qns + hard_qns
    random.shuffle(all_questions)
    return all_questions

# View to handle quiz display
def quiz_view(request):
    data = load_questions()
    if not data:
        return render(request, "quiz.html", {"error": "Error loading questions!"})

    if request.method == "POST":
        selected_category = request.POST.get("category")
        if selected_category not in data:
            return render(request, "quiz.html", {"error": "Invalid category selected!"})

        questions = get_random_questions(data, selected_category)
        request.session["questions"] = questions
        request.session["category"] = selected_category
        request.session["score"] = 0

        return redirect("quiz_questions")

    return render(request, "quiz.html", {"categories": list(data.keys())})

# View to handle question answering
def quiz_questions(request):
    questions = request.session.get("questions", [])
    current_question_index = request.session.get("current_question_index", 0)
    
    if current_question_index >= len(questions):
        return redirect("quiz_result")  # All questions have been answered, go to result page

    current_question = questions[current_question_index]
    
    # Handle form submission (answering a question)
    if request.method == "POST":
        user_answer = request.POST.get("answer")
        if user_answer:
            # Save the answer (you can add logic to store it in the session or database)
            answers = request.session.get("answers", [])
            answers.append({"question": current_question, "user_answer": user_answer})
            request.session["answers"] = answers

            # Move to next question
            current_question_index += 1
            request.session["current_question_index"] = current_question_index

            return redirect("quiz_questions")  # Show next question
    
    return render(request, "quiz_questions.html", {
        "question": current_question,
        "question_counter": current_question_index + 1,  # For displaying question number
        "proctored":True
    })


# View to show quiz result
def quiz_result(request):
    answers = request.session.get("answers", [])
    total_questions = len(answers)
    score = 0

    for answer in answers:
        # Compare the selected answer with the correct answer
        if answer["user_answer"] == answer["question"]["answer"]:
            score += 1

    percentage = (score / total_questions) * 100 if total_questions else 0

    # Clear session data after showing results (optional)
    request.session["answers"] = []
    request.session["current_question_index"] = 0

    return render(request, "result.html", {
        "score": score,
        "total": total_questions,
        "percentage": percentage
    })

