from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
import io
import SimpleITK as sitk
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pydantic import BaseModel
import requests
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import RecognitionHistory, SessionLocal, engine

app = FastAPI()

origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

YOLO_model = YOLO(r"D:\Code\Python Code\CT552\YOLO results\runs_yolov8x_512_new\train\weights\best.pt")

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

id2label = {0: 'lung_ct_2', 1: 'nodule'}

PREDICT_SLICE_PNG_DIR = Path("predict_image")
PREDICT_SLICE_PNG_DIR.mkdir(exist_ok=True)

def update_prediction_history(file_id, predicted_image, slice_index):
    session = SessionLocal()
    history_entry = session.query(RecognitionHistory).filter(RecognitionHistory.file_id == file_id, RecognitionHistory.slice_index.is_(None)).first()

    history_entries = session.query(RecognitionHistory).filter(
        RecognitionHistory.file_id == file_id,
        RecognitionHistory.slice_index.isnot(None)
    ).first()
    
    if history_entry:
        history_entry.predicted_image = predicted_image
        history_entry.slice_index = slice_index
        session.commit()
    else:
        new_history = RecognitionHistory(
            file_id=history_entries.file_id,
            file1_name=history_entries.file1_name,
            file2_name=history_entries.file2_name,
            extracted_images=history_entries.extracted_images,
            predicted_image=predicted_image,
            slice_index=slice_index
        )
        session.add(new_history)
        session.commit()
    session.close()

# Định nghĩa request model cho FastAPI
class ImageRequest(BaseModel):
    image_url: str
    
# Endpoint để truy cập ảnh .png
@app.get("/predict_image/{image_name}")
async def get_predict_image(image_name: str):
    image_path = Path(PREDICT_SLICE_PNG_DIR) / image_name
    if image_path.exists():
        return FileResponse(image_path)
    return {"error": "Image not found"}

# Endpoint để xử lý ảnh
@app.post("/predict/")
async def predict(image_request: ImageRequest):
    # Tải ảnh từ URL
    try:
        response = requests.get(image_request.image_url)
        print(image_request.image_url)
        response.raise_for_status()  # Kiểm tra lỗi
        image_data = io.BytesIO(response.content)
        image_array = np.frombuffer(image_data.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể tải ảnh từ URL: {e}")
    
    # for file in PREDICT_SLICE_PNG_DIR.glob("*.png"):
    #     file.unlink()
    
    box = YOLO_model(image)
    file_name = image_request.image_url.split("/")[-1].split(".png")[0]
    file_id = "_".join(file_name.split("_")[:-2])  # Lấy ID từ URL
    slice_index = int(file_name.split("_")[-1])
    output_path = f"{file_name}.jpg"
    
    if len(box[0].boxes.xyxy) != 0:
        x_min_yolo, y_min_yolo, x_max_yolo, y_max_yolo = box[0].boxes.xyxy[0].tolist()
        cv2.rectangle(image, (int(x_min_yolo), int(y_min_yolo)), (int(x_max_yolo), int(y_max_yolo)), color=(255,0,0), thickness=2)
        
    cv2.imwrite(f"./predict_image/{output_path}", image)
    predicted_image_url = f"http://localhost:8080/predict_image/{output_path}"
    update_prediction_history(file_id, predicted_image_url, slice_index)
    
    return {"predicted_image": predicted_image_url}