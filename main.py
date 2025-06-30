from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

app = FastAPI()
model = YOLO("best.pt")  
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    results = model(image)
    return {"predictions": results[0].boxes.data.tolist()}
