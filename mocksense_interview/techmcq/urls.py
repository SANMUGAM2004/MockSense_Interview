from django.urls import path
from .views import quiz_view

urlpatterns = [
    path('', quiz_view, name='mcq-home'),
]