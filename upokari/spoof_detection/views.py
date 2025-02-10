from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import cv2
import numpy as np
import os
import face_recognition
from django.shortcuts import HttpResponse
from .anti_spoof_predict import AntiSpoofPredict
from .generate_patches import CropImage
from .utility import parse_model_name

import requests
import io
from PIL import Image
from .models import SpoofAttempt
from dotenv import load_dotenv
load_dotenv()

class AntiSpoofingAndFacialRecognitionView(APIView):
    permission_classes = [IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        try:
            # Retrieve uploaded files
            reference_image_file = request.FILES.get('reference_image')
            target_image_file = request.FILES.get('target_image')

            if not reference_image_file or not target_image_file:
                return Response({"error": "Both reference and target images are required."}, status=400)
            
            if reference_image_file.size == 0 or target_image_file.size == 0:
                return Response({"error": "Uploaded images are empty."}, status=400)

            # Process the target image
            target_image_data = np.frombuffer(target_image_file.read(), np.uint8)
            target_image = cv2.imdecode(target_image_data, cv2.IMREAD_COLOR)

            # Perform anti-spoofing prediction
            label, value = self.predict_spoof(target_image)
            if label != 1:
                return Response({"result": "Spoof detected in the target image.", "score": value}, status=200)

            # Process reference image for facial recognition
            ref_image_data = np.frombuffer(reference_image_file.read(), np.uint8)
            reference_image = cv2.imdecode(ref_image_data, cv2.IMREAD_COLOR)
            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
          
            reference_encodings = face_recognition.face_encodings(reference_image)

            if len(reference_encodings) == 0:
                return Response({"error": "No face found in the reference image."}, status=400)

            reference_encoding = reference_encodings[0]

            # Perform facial recognition on the target image
            target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            target_encodings = face_recognition.face_encodings(target_image_rgb)
            target_locations = face_recognition.face_locations(target_image_rgb)

            if len(target_encodings) == 0:
                return Response({"error": "No face found in the target image."}, status=400)

            # Compare faces
            matches = face_recognition.compare_faces([reference_encoding], target_encodings[0], tolerance=0.4)
            face_distance = face_recognition.face_distance([reference_encoding], target_encodings[0])[0]
            match_percentage = round((1 - face_distance) * 100, 2)
            match_result = "Match" if matches[0] else "No Match"

            return Response({
                "anti_spoof_result": "Real",
                "match_result": match_result,
                "match_percentage": match_percentage
            }, status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=500)

    def predict_spoof(self, image):
        model_dir = "./resources/anti_spoof_models"  # Update this path as necessary
        device_id = 0
        model_test = AntiSpoofPredict(device_id)
        image_cropper = CropImage()
        image = cv2.resize(image, (int(image.shape[0] * 3 / 4), image.shape[0]))
        if not self.check_image(image):
            raise ValueError("Invalid image aspect ratio. Height/Width should be 4/3.")
        image_bbox = model_test.get_bbox(image)
        prediction = np.zeros((1, 3))
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))

        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        return label, value

    def check_image(self, image, tolerance=0.05):
        height, width, channel = image.shape
        aspect_ratio = width / height
        target_ratio = 3 / 4
        return abs(aspect_ratio - target_ratio) <= tolerance





SDK_URL = os.environ.get("SDK_URL")


from django.core.files.base import ContentFile


class IDDocumentLivenessView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get_client_ip(self, request):
        """Extract real IP address, ignoring local/private addresses"""
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            ip_list = x_forwarded_for.split(",")
            for ip in ip_list:
                ip = ip.strip()
                if not ip.startswith(("10.", "192.168.", "172.", "127.")):  # Ignore private/local IPs
                    return ip
        return request.META.get("REMOTE_ADDR", "Unknown")

    def get_location_from_ip(self, ip):
        """Fetch location details using IP lookup (Ignore local IPs)"""
        if ip.startswith(("127.", "10.", "192.168.", "172.")):  # Local network, no location
            return "Local Network"

        try:
            response = requests.get(f"http://ip-api.com/json/{ip}")
            if response.status_code == 200:
                data = response.json()
                return f"{data.get('city')}, {data.get('country')}"
        except Exception:
            return "Unknown Location"

    def post(self, request, *args, **kwargs):
        try:
            user_id = request.data.get("user_id")
            image_file = request.FILES.get("image")

            if not user_id or not image_file:
                return Response({"error": "User ID and image are required."}, status=400)

            # Convert image file to bytes
            image = Image.open(image_file)
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")
            img_bytes.seek(0)

            # Send request to SDK
            files = {"image": img_bytes}
            sdk_response = requests.post(SDK_URL, files=files)

            if not sdk_response.ok:
                return Response({"error": "Failed to process image with SDK"}, status=500)

            sdk_result = sdk_response.json()
            process_results = sdk_result.get("result", {})

            screenReply = process_results.get("screenReply", 1.0)
            portraitReplace = process_results.get("portraitReplace", 1.0)

            # Get client IP & location
            client_ip = self.get_client_ip(request)
            location = self.get_location_from_ip(client_ip)

            # Check if spoof detected
            is_spoof = screenReply < 0.5 or portraitReplace < 0.5

            if is_spoof:
                # Save image as proof
                spoof_instance = SpoofAttempt(user_id=user_id, ip_address=client_ip, location=location)
                spoof_instance.image.save(f"{user_id}_spoof.jpg", ContentFile(img_bytes.getvalue()))
                spoof_instance.save()

                return Response({
                    "result": "Spoof detected",
                    "user_id": user_id,
                    "screenReply": screenReply,
                    "portraitReplace": portraitReplace,
                    "ip_address": client_ip,
                    "location": location,
                    "image_url": request.build_absolute_uri(spoof_instance.image.url),
                }, status=200)

            return Response({
                "result": "Real document",
                "sdk_response": sdk_result
            }, status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=500)




from rest_framework.generics import ListAPIView
from django.utils.dateparse import parse_datetime
from .serializers import SpoofAttemptSerializer

class QuerySpoofAttemptsView(ListAPIView):
    serializer_class = SpoofAttemptSerializer

    def get_queryset(self):
        queryset = SpoofAttempt.objects.all()
        user_id = self.request.query_params.get("user_id")
        start_date = self.request.query_params.get("start_date")
        end_date = self.request.query_params.get("end_date")
        ip_address = self.request.query_params.get("ip")

        if user_id:
            queryset = queryset.filter(user_id=user_id)
        if start_date:
            queryset = queryset.filter(detected_at__gte=parse_datetime(start_date))
        if end_date:
            queryset = queryset.filter(detected_at__lte=parse_datetime(end_date))
        if ip_address:
            queryset = queryset.filter(ip_address=ip_address)

        return queryset
