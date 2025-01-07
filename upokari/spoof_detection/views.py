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
