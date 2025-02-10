
from django.urls import path
from . import views
# from .views import quiz_view, quiz_questions, quiz_result

urlpatterns = [
    path("", views.quiz_view, name="quiz_view"),
    path("quiz-questions/", views.quiz_questions, name="quiz_questions"),
    path("quiz-result/", views.quiz_result, name="quiz_result"),
]
