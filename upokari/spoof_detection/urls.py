# spoof_detection/urls.py
from django.urls import path
from .views import *
# from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('face_matching/', AntiSpoofingAndFacialRecognitionView.as_view(), name='anti_spoofing'),
    # path('token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    # path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path("id-document-liveness/", IDDocumentLivenessView.as_view(), name="id-document-liveness"),
    path("query-spoof-attempts/", QuerySpoofAttemptsView.as_view(), name="query-spoof-attempts"),
    path("get-image/", GetSpoofImageView.as_view(), name="get_spoof_image"),
]
