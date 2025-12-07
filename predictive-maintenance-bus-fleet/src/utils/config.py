
import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    PG_HOST = os.getenv("PG_HOST")
    PG_PORT = os.getenv("PG_PORT")
    PG_DB   = os.getenv("PG_DB")
    PG_USER = os.getenv("PG_USER")
    PG_PASS = os.getenv("PG_PASSWORD")

    MODELS_DIR = "models"
