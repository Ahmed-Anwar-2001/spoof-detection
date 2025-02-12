import mongoengine as me
from datetime import datetime
import re
import pytz

BD_TIMEZONE = pytz.timezone("Asia/Dhaka")
class IDSpoofAttempt(me.Document):
    user_id = me.StringField(max_length=255, required=True)
    detected_at = me.DateTimeField(default=lambda: datetime.now(BD_TIMEZONE))
    image = me.StringField()  # Store image file path as a string
    ip_address = me.StringField()  # Use StringField for IP address
    location = me.StringField(max_length=255, default="Unknown")
    
    meta = {
        'collection': 'spoof_attempts'  # Specifies the MongoDB collection name
    }

    def __str__(self):
        return f"Spoof - {self.user_id} at {self.detected_at} ({self.ip_address}, {self.location})"
    
    def clean(self):
        # Optional: Validate the IP address format using regex
        if self.ip_address:
            ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
            if not re.match(ip_pattern, self.ip_address):
                raise me.ValidationError("Invalid IP address format.")
