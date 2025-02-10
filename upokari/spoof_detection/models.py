from django.db import models

class SpoofAttempt(models.Model):
    user_id = models.CharField(max_length=255)
    detected_at = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to="spoof_attempts/")  # Store spoofed images
    ip_address = models.GenericIPAddressField()
    location = models.CharField(max_length=255, default="Unknown")

    def __str__(self):
        return f"Spoof - {self.user_id} at {self.detected_at} ({self.ip_address}, {self.location})"
