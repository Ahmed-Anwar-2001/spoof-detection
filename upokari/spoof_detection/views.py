from rest_framework.views import APIView
from rest_framework.response import Response
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

from db_connection import *

import requests
import io
from PIL import Image
from .models import IDSpoofAttempt
from datetime import datetime
import pytz

from .serializers import IDSpoofAttemptSerializer

from dotenv import load_dotenv
load_dotenv()

class AntiSpoofingAndFacialRecognitionView(APIView):
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
BD_TIMEZONE = pytz.timezone("Asia/Dhaka")
MAX_IMAGE_SIZE_KB = 100  # Target max size per image

class IDDocumentLivenessView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get_client_ip(self, request):
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        if x_forwarded_for:
            ip_list = x_forwarded_for.split(",")
            for ip in ip_list:
                ip = ip.strip()
                if not ip.startswith(("10.", "192.168.", "172.", "127.")):  # Ignore private/local IPs
                    return ip
        return request.META.get("REMOTE_ADDR", "Unknown")

    def get_location_from_ip(self, ip):
        if ip.startswith(("127.", "10.", "192.168.", "172.")):  # Local network, no location
            return "Local Network"
        try:
            response = requests.get(f"http://ip-api.com/json/{ip}")
            if response.status_code == 200:
                data = response.json()
                return f"{data.get('city')}, {data.get('country')}"
        except Exception:
            return "Unknown Location"

    def compress_image(self, image, target_size_kb=MAX_IMAGE_SIZE_KB):
        """Compress the image to fit within target_size_kb while maintaining quality."""
        img_bytes = io.BytesIO()
        quality = 85  # Start with high quality

        # Convert RGBA or grayscale to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if too large
        max_dim = 800  # Reduce to max 800px width/height
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim))

        while True:
            img_bytes.seek(0)
            image.save(img_bytes, format="JPEG", quality=quality, optimize=True)
            size_kb = len(img_bytes.getvalue()) / 1024  # Convert bytes to KB

            if size_kb <= target_size_kb or quality <= 30:
                break  # Stop if within size limit or quality too low
            quality -= 5  # Reduce quality and try again

        img_bytes.seek(0)
        return img_bytes

    def post(self, request, *args, **kwargs):
        try:
            user_id = request.data.get("user_id")
            image_file = request.FILES.get("image")

            if not user_id or not image_file:
                return Response({"error": "User ID and image are required."}, status=400)

            # Open image file
            image = Image.open(image_file)
            compressed_img_bytes = self.compress_image(image)

            # Send request to SDK
            files = {"image": compressed_img_bytes}
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

            # Get current Bangladesh time correctly
            now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)  # UTC time
            now_bd = now_utc.astimezone(BD_TIMEZONE)  # Convert to Bangladesh Time

            # Check if spoof detected
            is_spoof = screenReply < 0.5 or portraitReplace < 0.5

            if is_spoof:
                # Create folder structure for monthly organization
                month_folder = now_bd.strftime("%b").lower()  # e.g., "feb", "mar"
                base_folder = "spoof_attempts"
                save_dir = os.path.join("media", base_folder, month_folder)
                os.makedirs(save_dir, exist_ok=True)

                # Correct file name format: userID_YYYYMMDD_HHMMSS.jpg
                filename = f"{user_id}_{now_bd.strftime('%Y_%m_%d__%H_%M_%S')}.jpg"
                image_path = os.path.join(save_dir, filename)
                db_image_path = os.path.join(base_folder, month_folder, filename)

                # Save compressed image
                with open(image_path, "wb") as f:
                    f.write(compressed_img_bytes.getvalue())

                # Save spoof attempt to MongoDB
                spoof_data = {
                    "user_id": user_id,
                    "detected_at": now_bd,  # Correct BD time
                    "ip_address": client_ip,
                    "location": location,
                    "image": db_image_path
                }
                db['spoof_attempts'].insert_one(spoof_data)

                serializer = IDSpoofAttemptSerializer(spoof_data)

                return Response({
                    "result": "Spoof detected",
                    "user_id": user_id,
                    "screenReply": screenReply,
                    "portraitReplace": portraitReplace,
                    "ip_address": client_ip,
                    "location": location,
                    "image_url": db_image_path,
                    "detected_at": now_bd,

                }, status=200)
            else:
                return Response({
                    "result": "Real document",
                    "sdk_response": sdk_result
                }, status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=500)





from django.http import JsonResponse
from django.utils.dateparse import parse_datetime
from rest_framework.views import APIView
from rest_framework.response import Response
from pymongo import MongoClient
import os


spoof_attempts_collection = db["spoof_attempts"]

class QuerySpoofAttemptsView(APIView):
    def get(self, request, *args, **kwargs):
        """Fetch spoof attempts based on filters: user_id, date range, IP, location."""
        
        # Extract query params
        user_id = request.GET.get("user_id")
        start_date = request.GET.get("start_date")
        end_date = request.GET.get("end_date")
        ip_address = request.GET.get("ip")
        location = request.GET.get("location")

        # Build MongoDB query dynamically
        query = {}
        if user_id:
            query["user_id"] = user_id
        if ip_address:
            query["ip_address"] = ip_address
        if location:
            query["location"] = location
        if start_date:
            query["detected_at"] = {"$gte": parse_datetime(start_date)}
        if end_date:
            query.setdefault("detected_at", {})["$lte"] = parse_datetime(end_date)

        # Fetch data from MongoDB
        results = list(spoof_attempts_collection.find(query, {"_id": 0}))  # Exclude `_id` field

        if not results:
            return Response({"message": "No spoof attempts found"}, status=404)

        return Response({"data": results}, status=200)



import os
from django.http import FileResponse
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response

class GetSpoofImageView(APIView):
    def get(self, request):
        month = request.query_params.get("month")  # e.g., "feb"
        filename = request.query_params.get("filename")  # e.g., "sdvsvdvd_2025_02_13__03_27_20.jpg"

        if not month or not filename:
            return Response({"error": "Both 'month' and 'filename' are required"}, status=400)

        # Construct full absolute path
        file_path = os.path.join(settings.BASE_DIR, "media", "spoof_attempts", month, filename)

        # Debugging - Print the constructed path
        print(f"Looking for file at: {file_path}")

        # Check if file exists
        if os.path.exists(file_path):
            return FileResponse(open(file_path, "rb"), content_type="image/jpeg")

        return Response({"error": "Image not found"}, status=404)

