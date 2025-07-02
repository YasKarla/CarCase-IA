from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from io import BytesIO
from PIL import Image, UnidentifiedImageError

app = FastAPI()

try:
    model = YOLO("best.pt")
except Exception as e:
    raise RuntimeError(f"Erro ao carregar modelo: {str(e)}")

@app.get("/")
def root():
    return {"message": "API YOLO ativa. Envie uma imagem para /predict/ via POST."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="Arquivo vazio.")

        image = Image.open(BytesIO(contents))

        results = model(image)
        predictions = []

        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names.get(class_id, "Classe desconhecida")
            confidence = float(box.conf[0].item())
            coords = box.xyxy[0].tolist()

            predictions.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "box": coords
            })

        return {"predictions": predictions}

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Arquivo enviado não é uma imagem válida.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
