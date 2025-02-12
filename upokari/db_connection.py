import os
from pymongo import MongoClient
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

# Retrieve connection details from environment variables
db_name = os.environ.get("MONGO_DB")
user = os.environ.get("MONGO_USER")
password = os.environ.get("MONGO_PASSWORD")
mongo_host = os.environ.get("MONGO_HOST", "mongo")  # Use 'mongo' for Docker

# URL encode credentials
encoded_user = quote_plus(user)
encoded_password = quote_plus(password)

# Build the MongoDB connection URI
uri = f"mongodb://{encoded_user}:{encoded_password}@{mongo_host}:27017/{db_name}?authSource=admin"

# Connect to MongoDB
client = MongoClient(uri)

# Access the database
db = client[db_name]
