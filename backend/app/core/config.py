import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

DATA_DIR      = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR  = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")   # ← ajouté