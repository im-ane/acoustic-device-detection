from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.pipeline import analyze_audio

router = APIRouter()


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    # if not file.filename.endswith(".wav"):
    #     raise HTTPException(status_code=400, detail="File must be WAV")

    result = analyze_audio(file)

    return result