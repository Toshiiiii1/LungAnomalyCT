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

app = FastAPI()

# Thư mục lưu trữ file tải lên và kết quả
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    MAX_BOUND = 400.0    # Giá trị tối đa để chuẩn hóa ảnh
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
    for i, slice_array in enumerate(image_array):
        img = normalize(slice_array)  # Chuẩn hóa lát cắt ảnh
        img_grey = img * 255  # Chuyển đổi ảnh chuẩn hóa sang ảnh xám (giá trị từ 0-255)
        img_rgb = np.stack((img_grey,) * 3, -1)  # Tạo ảnh RGB từ ảnh xám
        png_path = OUTPUT_DIR / f"{id[:-4]}_slice_{i}.png"  # Đường dẫn lưu file .png
        cv2.imwrite(png_path, img_rgb)  # Lưu ảnh dưới dạng file .png
        png_files.append(str(png_path))  # Thêm đường dẫn file .png vào danh sách

    return png_files  # Trả về danh sách đường dẫn các file .png đã lưu

def save_extraction_history(file1_name, file2_name, file_id, extracted_images):
    session = SessionLocal()
    history_entry = RecognitionHistory(
        file_id=file_id,
        file1_name=file1_name,
        file2_name=file2_name,
        extracted_images=json.dumps(extracted_images)
    )
    session.add(history_entry)
    session.commit()
    session.close()


@app.post("/upload")
async def upload_files(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    # Lưu file tải lên
    file1_path = Path(UPLOAD_DIR) / file1.filename
    file2_path = Path(UPLOAD_DIR) / file2.filename

    with open(file1_path, "wb") as f:
        f.write(await file1.read())

    with open(file2_path, "wb") as f:
        f.write(await file2.read())

    # Xử lý file với SimpleITK
    result_files = process_files(file1_path, file2_path, file1.filename)
    image_urls = [f"http://localhost:8000/output/{Path(file).name}" for file in result_files]
    
    file_id = file1.filename[:-4]  # Sử dụng tên file `.mhd` làm ID
     # Lưu lịch sử trích xuất
    save_extraction_history(file1.filename, file2.filename, file_id, image_urls)

    # Trả về danh sách tên file
    return {"files": image_urls}

# Endpoint để truy cập ảnh .png
@app.get("/output/{image_name}")
async def get_image(image_name: str):
    image_path = Path(OUTPUT_DIR) / image_name
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
            extracted_images = json.loads(record.extracted_images) if record.extracted_images else []
        except json.JSONDecodeError:
            extracted_images = []

        history.append({
            "id": record.id,
            "file_id": record.file_id,
            "file1_name": record.file1_name,
            "file2_name": record.file2_name,
            "extracted_images": extracted_images,
            "predicted_image": record.predicted_image,
            "slice_index": record.slice_index,
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
        mhd_file = Path(UPLOAD_DIR) / record.file1_name
        raw_file = Path(UPLOAD_DIR) / record.file2_name
        
        if mhd_file.exists():
            mhd_file.unlink()
        if raw_file.exists():
            raw_file.unlink()

        # Xóa các file ảnh trong thư mục output
        output_files = Path(OUTPUT_DIR).glob(f"{mhd_file.stem}_*.png")
        for output_file in output_files:
            if output_file.exists():
                output_file.unlink()
                
        # Thư mục chứa các file ảnh .jpg
        predict_image_dir = Path(r"D:\Code\Python Code\CT552\demo\model_server\predict_image")

        # Tìm tất cả các file .jpg trong thư mục predict_image_dir
        output_files = predict_image_dir.glob("*.jpg")

        # Xóa các file .jpg nếu chúng tồn tại
        for output_file in output_files:
            if output_file.exists():
                output_file.unlink()

    # Xóa bản ghi trong cơ sở dữ liệu
    db.delete(record)
    db.commit()

    return {"message": "History record deleted successfully"}