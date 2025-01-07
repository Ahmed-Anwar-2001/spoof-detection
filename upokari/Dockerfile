# Use the official Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app



# Copy the requirements file from the upokari folder
COPY upokari/requirements.txt /app/

# Copy the libs folder to the container
COPY upokari/libs /app/libs

# Install Python dependencies, including dlib
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Django project into the container
COPY upokari /app/

# Expose the port that Django runs on
EXPOSE 8000

# Default command to run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
