# from django.urls import path
# from .views import discussion_page, group_discussion,detect_emotion  # ✅ Import your views

# urlpatterns = [
#     path('', discussion_page, name="group_discussion_page"),  # Renders the discussion page
#     path('api/', group_discussion, name="group_discussion_api"),  # Handles AI responses
#     path('detect_emotion/', detect_emotion, name='detect_emotion'),
# ]
from django.urls import path
from .views import discussion_view, detect_emotion, group_discussion

urlpatterns = [
    path("", discussion_view, name="discussion"),
    path("detect_emotion/", detect_emotion, name="detect_emotion"),
    # path("group_discussion/", group_discussion, name="group_discussion"),
    path("gd/", group_discussion, name="group_discussion"),
]
# from django.urls import path
# from .views import detect_emotion, discussion_view, group_discussion

# urlpatterns = [
#     path("detect_emotion/", detect_emotion, name="detect_emotion"),
#     path("discussion/", discussion_view, name="discussion_view"),
#     path("gd/api/", group_discussion, name="group_discussion"),
# ]
