from django.contrib import admin
from django.urls import path, include
from . import views
urlpatterns = [
    path('predict/', views.PredictImageUploadAPIView.as_view()),
    path('deblur/',views.DeblurImageAPIView.as_view())
]