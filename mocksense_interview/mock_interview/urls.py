from django.urls import path
from . import views

# urlpatterns = [
    # path("", views.index, name="mock_interview_index"),  # Home page or dashboard
    # path("quiz/", views.quiz_view, name="mock_interview_quiz"),  # Quiz page
    # path("start/", views.start, name="start_interview"),  # Start interview
    # path("result/", views.result, name="mock_interview_result"),  # Result page
urlpatterns = [
    path('start/', views.start_interview, name='start_interview'),
    path('quiz/', views.quiz_view, name='mock_interview_quiz'),
    path('video_feed/', views.video_feed, name='mock_interview_video_feed'),
    path('listen/', views.listen_answer, name='mock_interview_listen'),
    path('ask_question/<int:question_index>/', views.ask_questions, name='ask_question'),
    path('ask_question/', views.ask_questions, name='ask_question'),
    path('result/', views.mock_interview_result, name='mock_interview_result'),  # Add this line
]

# ]
