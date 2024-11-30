from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
import os
import SimpleITK as sitk
from pathlib import Path
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from fastapi.responses import FileResponse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import SessionLocal, RecognitionHistory
import json
from sqlalchemy.orm import Session
from ultralytics import YOLO

app = FastAPI()

# Thư mục lưu trữ file tải lên và kết quả
MHD_RAW_UPLOAD_DIR = Path("uploads_mhd_raw")
MHD_RAW_OUTPUT_DIR = Path("output_mhd_raw")
MHD_RAW_PREDICT_SLICE_PNG_DIR = Path("predict_image_mhd_raw")
PNG_UPLOAD_DIR = Path("uploads_png")
PNG_OUTPUT_DIR = Path("output_png")
PNG_PREDICT_SLICE_PNG_DIR = Path("predict_image_png")

os.makedirs(MHD_RAW_UPLOAD_DIR, exist_ok=True)
os.makedirs(MHD_RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(MHD_RAW_PREDICT_SLICE_PNG_DIR, exist_ok=True)
os.makedirs(PNG_UPLOAD_DIR, exist_ok=True)
os.makedirs(PNG_OUTPUT_DIR, exist_ok=True)
os.makedirs(PNG_PREDICT_SLICE_PNG_DIR, exist_ok=True)

YOLO_model = YOLO(r"D:\Code\Python Code\CT552\YOLO results\runs_yolov8x_512_new\train\weights\best.pt")

# CORS handle
origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def normalize(image):
    MIN_BOUND = -1000.0  # Giá trị tối thiểu để chuẩn hóa ảnh
    MAX_BOUND = 1000.0    # Giá trị tối đa để chuẩn hóa ảnh
    # Chuẩn hóa ảnh theo công thức (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    # Đặt giá trị lớn hơn 1 thành 1 (giới hạn trên)
    image[image > 1] = 1.
    # Đặt giá trị nhỏ hơn 0 thành 0 (giới hạn dưới)
    image[image < 0] = 0.
    return image  # Trả về ảnh đã được chuẩn hóa

def process_files(mhd_file: Path, raw_file: Path, id) -> List[str]:
    # Đọc ảnh y khoa từ file .mhd và .raw
    itk_image = sitk.ReadImage(str(mhd_file))
    # Chuyển đổi ảnh từ định dạng ITK sang array numpy
    image_array = sitk.GetArrayFromImage(itk_image)

    # Xóa tất cả các file ảnh cũ có định dạng .png nếu tồn tại trong thư mục PNG_DIR
    # for file in OUTPUT_DIR.glob("*.png"):
    #     file.unlink()

    # Lưu từng lát cắt của ảnh dưới dạng file .png
    png_files = []  # Danh sách để lưu đường dẫn các file .png đã lưu
    pred_img_files = []
    for i, slice_array in enumerate(image_array):
        img = normalize(slice_array)  # Chuẩn hóa lát cắt ảnh
        img_grey = img * 255  # Chuyển đổi ảnh chuẩn hóa sang ảnh xám (giá trị từ 0-255)
        img_rgb = np.stack((img_grey,) * 3, -1)  # Tạo ảnh RGB từ ảnh xám
        png_path = MHD_RAW_OUTPUT_DIR / f"{id[:-4]}_slice_{i}.png"  # Đường dẫn lưu file .png
        cv2.imwrite(png_path, img_rgb)  # Lưu ảnh dưới dạng file .png
        png_files.append(str(png_path))  # Thêm đường dẫn file .png vào danh sách
        
        box = YOLO_model(img_rgb)
        
        if len(box[0].boxes.xyxy) != 0:
            for box in box[0].boxes.xyxy:
                x_min_yolo, y_min_yolo, x_max_yolo, y_max_yolo = box.tolist()
                cv2.rectangle(img_rgb, (int(x_min_yolo), int(y_min_yolo)), (int(x_max_yolo), int(y_max_yolo)), color=(0,0,255), thickness=2)
            pred_img_path = MHD_RAW_PREDICT_SLICE_PNG_DIR / f"{id[:-4]}_slice_{i}.png"
            cv2.imwrite(pred_img_path, img_rgb)
            pred_img_files.append(str(pred_img_path))
        else:
            continue

    return png_files, pred_img_files  # Trả về danh sách đường dẫn các file .png đã lưu

def save_extraction_history(file_name, file_type, original_images, predicted_images):
    session = SessionLocal()
    history_entry = RecognitionHistory(
        file_name=file_name,
        file_type=file_type,
        original_images=json.dumps(original_images),
        predicted_images=json.dumps(predicted_images)
    )
    session.add(history_entry)
    session.commit()
    session.close()

@app.post("/upload_mhd_raw")
async def upload_files(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Lưu file tải lên
    file1_path = Path(MHD_RAW_UPLOAD_DIR) / file1.filename
    file2_path = Path(MHD_RAW_UPLOAD_DIR) / file2.filename

    with open(file1_path, "wb") as f:
        f.write(await file1.read())

    with open(file2_path, "wb") as f:
        f.write(await file2.read())

    # Xử lý file với SimpleITK
    result_files, pred_files = process_files(file1_path, file2_path, file1.filename)
    image_urls = [f"http://localhost:8000/output_mhd_raw/{Path(file).name}" for file in result_files]
    pred_image_urls = [f"http://localhost:8000/predict_image_mhd_raw/{Path(file).name}" for file in pred_files]
    
    file_id = file1.filename[:-4]  # Sử dụng tên file `.mhd` làm ID
     # Lưu lịch sử trích xuất
    save_extraction_history(file_id, "mhd/raw", image_urls, pred_image_urls)

    # Trả về danh sách tên file
    return {"files": image_urls, "pred_files": pred_image_urls}

# Endpoint để truy cập ảnh .png
@app.get("/output_mhd_raw/{image_name}")
async def get_image(image_name: str):
    image_path = Path(MHD_RAW_OUTPUT_DIR) / image_name
    if image_path.exists():
        return FileResponse(image_path)
    return {"error": "Image not found"}

# Endpoint để truy cập ảnh .png
@app.get("/predict_image_mhd_raw/{image_name}")
async def get_predict_image(image_name: str):
    image_path = Path(MHD_RAW_PREDICT_SLICE_PNG_DIR) / image_name
    if image_path.exists():
        return FileResponse(image_path)
    return {"error": "Image not found"}

@app.post("/upload_png_jpg_jpeg")
async def upload_png_jp_file(file: UploadFile = File(...)):
     # Lưu file tải lên
    file_path = Path(PNG_UPLOAD_DIR) / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    img = cv2.imread(file_path)
    # img = cv2.resize(img, (512, 512))
    box = YOLO_model(img)
    if len(box[0].boxes.xyxy) != 0:
        x_min_yolo, y_min_yolo, x_max_yolo, y_max_yolo = box[0].boxes.xyxy[0].tolist()
        cv2.rectangle(img, (int(x_min_yolo), int(y_min_yolo)), (int(x_max_yolo), int(y_max_yolo)), color=(0,0,255), thickness=2)
    output_path = f"{file.filename.rsplit('.', 1)[0]}.png"
    pred_img_path = PNG_PREDICT_SLICE_PNG_DIR / output_path
    cv2.imwrite(pred_img_path, img)
    predicted_image_url = f"http://localhost:8000/predict_image_png/{output_path}"
    upload_image_url = f"http://localhost:8000/uploads_png/{file.filename}"
    
    save_extraction_history(file.filename[:-4], "png/jpg/jpeg", [upload_image_url], [predicted_image_url])
    
    return {
        "original_image": upload_image_url,
        "predicted_image": predicted_image_url
    }
    
@app.get("/uploads_png/{image_name}")
async def get_predict_image(image_name: str):
    image_path = Path(PNG_UPLOAD_DIR) / image_name
    if image_path.exists():
        return FileResponse(image_path)
    return {"error": "Image not found"}
    
@app.get("/predict_image_png/{image_name}")
async def get_predict_image(image_name: str):
    image_path = Path(PNG_PREDICT_SLICE_PNG_DIR) / image_name
    if image_path.exists():
        return FileResponse(image_path)
    return {"error": "Image not found"}

@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    records = db.query(RecognitionHistory).all()
    history = []

    for record in records:
        # Lấy thông tin từ từng bản ghi
        try:
            original_images = json.loads(record.original_images) if record.original_images else []
            predicted_images = json.loads(record.predicted_images) if record.predicted_images else []
        except json.JSONDecodeError:
            original_images = []
            predicted_images = []

        history.append({
            "id": record.id,
            "file_name": record.file_name,
            "file_type": record.file_type,
            "original_images": original_images,
            "predicted_images": predicted_images,
            "timestamp": record.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        })

    return {"history": history}

@app.delete("/history/{record_id}")
def delete_history(record_id: int, db: Session = Depends(get_db)):
    record = db.query(RecognitionHistory).filter(RecognitionHistory.id == record_id).first()
    history_entries = db.query(RecognitionHistory).filter(
        RecognitionHistory.file_id == record.file_id
    ).all()
    if not record:
        raise HTTPException(status_code=404, detail="History record not found")

    if len(history_entries) == 1:
        # Xóa file trong thư mục uploads
        mhd_file = Path(MHD_RAW_UPLOAD_DIR) / record.file1_name
        raw_file = Path(MHD_RAW_UPLOAD_DIR) / record.file2_name
        
        if mhd_file.exists():
            mhd_file.unlink()
        if raw_file.exists():
            raw_file.unlink()

        # Xóa các file ảnh trong thư mục output
        output_files = Path(MHD_RAW_OUTPUT_DIR).glob(f"{mhd_file.stem}_*.png")
        for output_file in output_files:
            if output_file.exists():
                output_file.unlink()

        # Tìm tất cả các file .jpg trong thư mục predict_image_dir
        output_files = Path(MHD_RAW_PREDICT_SLICE_PNG_DIR).glob("*.png")

        # Xóa các file .jpg nếu chúng tồn tại
        for output_file in output_files:
            if output_file.exists():
                output_file.unlink()

    # Xóa bản ghi trong cơ sở dữ liệu
    db.delete(record)
    db.commit()

    return {"message": "History record deleted successfully"}