import pytz
from rest_framework import serializers
from .models import IDSpoofAttempt

BD_TIMEZONE = pytz.timezone("Asia/Dhaka")

class IDSpoofAttemptSerializer(serializers.Serializer):
    user_id = serializers.CharField(max_length=255)
    detected_at = serializers.DateTimeField()
    image = serializers.CharField()
    ip_address = serializers.CharField(max_length=255)
    location = serializers.CharField(max_length=255)

    def to_representation(self, instance):
        """ Convert UTC to BD Time when returning data """
        data = super().to_representation(instance)
        detected_at_utc = data.get("detected_at")

        if detected_at_utc:
            detected_at_bd = detected_at_utc.astimezone(BD_TIMEZONE)
            data["detected_at"] = detected_at_bd.strftime("%Y-%m-%d %H:%M:%S %z")  # Formatted BD time

        return data


    def validate_ip_address(self, value):
        # Validate IP address format using regex
        ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
        if not re.match(ip_pattern, value):
            raise serializers.ValidationError("Invalid IP address format.")
        return value
