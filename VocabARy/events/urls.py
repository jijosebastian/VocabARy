from django.urls import path
from . import views

urlpatterns = [
    # General Page Routes
    path('', views.index, name='index'),
    path('detect/', views.detection, name='detection'),
    path('list/', views.word_list, name='word_list'),

    # Real-Time Video and Detection Stream
    path('video_feed/', views.video_feed, name='video_feed'),

    # API Endpoint for Fetching RAW Detected Object Names
    path('get_detected_objects/', views.get_detected_objects, name='get_detected_objects'),

    # Translation Routes
    path('translate/<str:object_name>/', views.translate_selected_object, name='translate_selected_object'),

    # Language and Learning APIs
    path('update_score_and_learn/', views.update_score_and_learn, name='update_score_and_learn'),
    path('api/set_language/', views.set_target_language, name='set_target_language'),
    path('api/objects/', views.get_detected_objects, name='get_detected_objects'),

    # 🆕 Gesture Writing Feature
    path('gesture_write/<str:object_name>/', views.gesture_write, name='gesture_write'),
    path('check_gesture/', views.check_gesture, name='check_gesture'),

    # 🆕 Handwriting (Air Writing) Demo
    # events/urls.py
    path('handwriting/', views.handwriting_test, name='handwriting'),
  # <--- new
    # If you want to render the handwriting page:
    path('handwriting_feed/', views.handwriting_test, name='handwriting_feed'),

]
