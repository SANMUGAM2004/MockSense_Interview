from django.urls import path
from . import views

# urlpatterns = [
    # path("", views.index, name="mock_interview_index"),  # Home page or dashboard
    # path("quiz/", views.quiz_view, name="mock_interview_quiz"),  # Quiz page
    # path("start/", views.start, name="start_interview"),  # Start interview
    # path("result/", views.result, name="mock_interview_result"),  # Result page
urlpatterns = [
    path("", views.index, name="mock_interview_index"),
    path('start/', views.start_interview, name='start_interview'),
    path('quiz/', views.quiz_view, name='mock_interview_quiz'),
    path('video_feed/', views.video_feed, name='mock_interview_video_feed'),
     path("save_answer/", views.save_answer, name="save_answer"),
    path('listen/', views.listen_answer, name='mock_interview_listen'),
    path('result/', views.mock_interview_result, name='mock_interview_result'),  # Add this line
    path("upload_resume/", views.upload_resume, name="upload_resume"),
    path("resume_uploading/", views.resume_uploading, name="resume_uploading"),
    path("limit-exceeded/", views.limit_exceeded, name="limit_exceeded"),

    
]

# ]
