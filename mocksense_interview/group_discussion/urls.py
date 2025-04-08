# from django.urls import path
# from .views import discussion_page, group_discussion,detect_emotion  # ✅ Import your views

# urlpatterns = [
#     path('', discussion_page, name="group_discussion_page"),  # Renders the discussion page
#     path('api/', group_discussion, name="group_discussion_api"),  # Handles AI responses
#     path('detect_emotion/', detect_emotion, name='detect_emotion'),
# ]
from django.urls import path
from .views import discussion_view, detect_emotion, group_discussion,start_grp_discussion_view,group_discussion_view,improve_sentences_view,save_user_gd_response

urlpatterns = [
    path("", discussion_view, name="discussion"),
    path("start", start_grp_discussion_view, name="start_group_discussion"),
    path("detect_emotion/", detect_emotion, name="detect_emotion"),
    path("disscussion", group_discussion_view, name="group_disscussion_view"),
    path('save_response/', save_user_gd_response, name='save_user_gd_response'),
    # path('save_response/', save_response, name='save_response'),
    # path('improve/', improve_sentences_view, name='improve_sentences'),
    path('result/', improve_sentences_view, name='gd_result'),

    # path("group_discussion/", group_discussion, name="group_discussion"),
    path("gd/", group_discussion, name="group_discussion"),
]
