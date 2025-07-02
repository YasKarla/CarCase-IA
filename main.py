from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

app = FastAPI()
model = YOLO("best.pt")  

@app.get("/")
def root():
    return {"message": "API YOLO ativa. Envie uma imagem para /predict/ via POST, testeeee."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    results = model(image)
    predictions = []

    for box in results[0].boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        confidence = float(box.conf[0].item())
        coords = box.xyxy[0].tolist()

        predictions.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "box": coords
        })

    return {"predictions": predictions}
