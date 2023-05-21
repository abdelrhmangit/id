import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from utils import classify_image

app = FastAPI(
    title="AI Image Classification API",
    description="""An API for classifying Images into AI and Human""",
)



@app.get("/", response_class=PlainTextResponse, tags=["home"])
async def home():
    note = """
   AI Image Detection API 🖼️\nAn API for detecting AI Generated Images AI\nNote: add \"/redoc\" to get the complete documentation.
    """
    return note

@app.post("/classify-image")
async def get_image(file: UploadFile = File(...)):

    contents = io.BytesIO(await file.read())
    file_bytes = np.asarray(bytearray(contents.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    cv2.imwrite("images.jpg", img)
    try:
        data = classify_image("images.jpg")
        return data
    except ValueError as e:
        e = "Error! Please upload a valid image type."
        return e