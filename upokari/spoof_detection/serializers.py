from rest_framework import serializers
from .models import SpoofAttempt

class SpoofAttemptSerializer(serializers.ModelSerializer):
    class Meta:
        model = SpoofAttempt
        fields = ["user_id", "detected_at", "image", "ip_address", "location"]
