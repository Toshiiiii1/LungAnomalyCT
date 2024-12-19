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
import pydicom
from io import BytesIO
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

app = FastAPI()

# Thư mục lưu trữ file tải lên và kết quả
MHD_RAW_UPLOAD_DIR = Path("uploads_mhd_raw")
MHD_RAW_OUTPUT_DIR = Path("output_mhd_raw")
MHD_RAW_PREDICT_SLICE_PNG_DIR = Path("predict_image_mhd_raw")
PNG_UPLOAD_DIR = Path("uploads_png")
PNG_OUTPUT_DIR = Path("output_png")
PNG_PREDICT_SLICE_PNG_DIR = Path("predict_image_png")
DCM_UPLOAD_DIR = Path("uploads_dcm")
DCM_OUTPUT_DIR = Path("output_dcm")
DCM_PREDICT_SLICE_PNG_DIR = Path("predict_image_dcm")

os.makedirs(MHD_RAW_UPLOAD_DIR, exist_ok=True)
os.makedirs(MHD_RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(MHD_RAW_PREDICT_SLICE_PNG_DIR, exist_ok=True)
os.makedirs(PNG_UPLOAD_DIR, exist_ok=True)
os.makedirs(PNG_OUTPUT_DIR, exist_ok=True)
os.makedirs(PNG_PREDICT_SLICE_PNG_DIR, exist_ok=True)
os.makedirs(DCM_UPLOAD_DIR, exist_ok=True)
os.makedirs(DCM_OUTPUT_DIR, exist_ok=True)
os.makedirs(DCM_PREDICT_SLICE_PNG_DIR, exist_ok=True)

# YOLO_model = YOLO(r"D:\Code\Python Code\CT552\YOLO results\runs_yolov8x_512_new\train\weights\best.pt")

FRCNN_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = FRCNN_model.roi_heads.box_predictor.cls_score.in_features
FRCNN_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
checkpoint = torch.load(r"D:\Code\Python Code\CT552\FRCNN result\frcnn_model_3_512.pth", map_location=torch.device('cpu'))
FRCNN_model.load_state_dict(checkpoint['model_state_dict'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRCNN_model.to(device)
FRCNN_model.eval()

# CORS handle
origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def apply_nms(orig_prediction, iou_thresh=0.3):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def normalize(image, file_type):
    MIN_BOUND = -1000.0  # Giá trị tối thiểu để chuẩn hóa ảnh
    if file_type == "mhd/raw":
        MAX_BOUND = 400.0    # Giá trị tối đa để chuẩn hóa ảnh
    else:
        MAX_BOUND = 1600.0
    # Chuẩn hóa ảnh theo công thức (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    # Đặt giá trị lớn hơn 1 thành 1 (giới hạn trên)
    image[image > 1] = 1.
    # Đặt giá trị nhỏ hơn 0 thành 0 (giới hạn dưới)
    image[image < 0] = 0.
    return image  # Trả về ảnh đã được chuẩn hóa

def get_versioned_filename(file_path):
    base = file_path.stem  # Lấy phần tên file (không có đuôi)
    suffix = file_path.suffix  # Lấy phần đuôi file
    i = 1
    while file_path.exists():
        file_path = file_path.with_name(f"{base}_v{i}{suffix}")
        i += 1
    return file_path

def process_mhd_raw_files(mhd_file: Path, raw_file: Path, id) -> List[str]:
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
    for i, slice_array in tqdm(enumerate(image_array)):
        img = normalize(slice_array, "mhd/raw")  # Chuẩn hóa lát cắt ảnh
        img_grey = img * 255  # Chuyển đổi ảnh chuẩn hóa sang ảnh xám (giá trị từ 0-255)
        img_rgb = np.stack((img_grey,) * 3, -1)  # Tạo ảnh RGB từ ảnh xám
        # png_path = MHD_RAW_OUTPUT_DIR / f"{id[:-4]}_slice_{i}.png"  # Đường dẫn lưu file .png
        png_path = MHD_RAW_OUTPUT_DIR / f"{id}_slice_{i}.png"  # Đường dẫn lưu file .png
        cv2.imwrite(png_path, img_rgb)  # Lưu ảnh dưới dạng file .png
        png_files.append(str(png_path))  # Thêm đường dẫn file .png vào danh sách
        
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1))
        img_tensor = img_tensor.float().div(255)
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            prediction = FRCNN_model([img_tensor])[0]
        prediction = apply_nms(prediction)
        score_threshold = 0.25
        filtered_indices = prediction["scores"] > score_threshold
        filtered_boxes = prediction["boxes"][filtered_indices]
        filtered_labels = prediction["labels"][filtered_indices]
        filtered_scores = prediction["scores"][filtered_indices]
        
        filtered_data = {
            "boxes": filtered_boxes,
            "labels": filtered_labels,
            "scores": filtered_scores
        }
        result = filtered_data["boxes"]
        if len(result) != 0:
            for box in result:
                x_min_frcnn, y_min_frcnn, x_max_frcnn, y_max_frcnn = box.tolist()
                cv2.rectangle(img_rgb, (int(x_min_frcnn), int(y_min_frcnn)), (int(x_max_frcnn), int(y_max_frcnn)), color=(0,0,255), thickness=2)
            # pred_img_path = MHD_RAW_PREDICT_SLICE_PNG_DIR / f"{id[:-4]}_slice_{i}.png"
            pred_img_path = MHD_RAW_PREDICT_SLICE_PNG_DIR / f"{id}_slice_{i}.png"
            cv2.imwrite(pred_img_path, img_rgb)
            pred_img_files.append(str(pred_img_path))
        else:
            continue
        
        # box = YOLO_model(img_rgb)
        
        # if len(box[0].boxes.xyxy) != 0:
        #     for box in box[0].boxes.xyxy:
        #         x_min_yolo, y_min_yolo, x_max_yolo, y_max_yolo = box.tolist()
        #         cv2.rectangle(img_rgb, (int(x_min_yolo), int(y_min_yolo)), (int(x_max_yolo), int(y_max_yolo)), color=(0,0,255), thickness=2)
        #     pred_img_path = MHD_RAW_PREDICT_SLICE_PNG_DIR / f"{id[:-4]}_slice_{i}.png"
        #     cv2.imwrite(pred_img_path, img_rgb)
        #     pred_img_files.append(str(pred_img_path))
        # else:
        #     continue

    return png_files, pred_img_files  # Trả về danh sách đường dẫn các file .png đã lưu

def process_dcm_files(dcm_files, folder_name) -> List[str]:
    # Lưu từng lát cắt của ảnh dưới dạng file .png
    png_files = []  # Danh sách để lưu đường dẫn các file .png đã lưu
    pred_img_files = []
    
    for i, dcm_file in tqdm(enumerate(dcm_files)):
        dcm = pydicom.dcmread(dcm_file)
        series_uid = dcm.SeriesInstanceUID
        slice_array = dcm.pixel_array
        img = normalize(slice_array, "dcm")  # Chuẩn hóa lát cắt ảnh
        img_grey = img * 255  # Chuyển đổi ảnh chuẩn hóa sang ảnh xám (giá trị từ 0-255)
        img_rgb = np.stack((img_grey,) * 3, -1)  # Tạo ảnh RGB từ ảnh xám
        png_path = DCM_OUTPUT_DIR / f"{folder_name}_slice_{i}.png"  # Đường dẫn lưu file .png
        cv2.imwrite(png_path, img_rgb)  # Lưu ảnh dưới dạng file .png
        png_files.append(str(png_path))  # Thêm đường dẫn file .png vào danh sách
        
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1))
        img_tensor = img_tensor.float().div(255)
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            prediction = FRCNN_model([img_tensor])[0]
        prediction = apply_nms(prediction)
        
        score_threshold = 0.25
        filtered_indices = prediction["scores"] > score_threshold
        filtered_boxes = prediction["boxes"][filtered_indices]
        filtered_labels = prediction["labels"][filtered_indices]
        filtered_scores = prediction["scores"][filtered_indices]
        
        filtered_data = {
            "boxes": filtered_boxes,
            "labels": filtered_labels,
            "scores": filtered_scores
        }
        result = filtered_data["boxes"]
        if len(result) != 0:
            for box in result:
                x_min_frcnn, y_min_frcnn, x_max_frcnn, y_max_frcnn = box.tolist()
                cv2.rectangle(img_rgb, (int(x_min_frcnn), int(y_min_frcnn)), (int(x_max_frcnn), int(y_max_frcnn)), color=(0,0,255), thickness=2)
            pred_img_path = DCM_PREDICT_SLICE_PNG_DIR / f"{folder_name}_slice_{i}.png"
            cv2.imwrite(pred_img_path, img_rgb)
            pred_img_files.append(str(pred_img_path))
        else:
            continue
        
        # box = YOLO_model(img_rgb)
        
        # if len(box[0].boxes.xyxy) != 0:
        #     for box in box[0].boxes.xyxy:
        #         x_min_yolo, y_min_yolo, x_max_yolo, y_max_yolo = box.tolist()
        #         cv2.rectangle(img_rgb, (int(x_min_yolo), int(y_min_yolo)), (int(x_max_yolo), int(y_max_yolo)), color=(0,0,255), thickness=2)
        #     pred_img_path = DCM_PREDICT_SLICE_PNG_DIR / f"{series_uid}_slice_{i}.png"
        #     cv2.imwrite(pred_img_path, img_rgb)
        #     pred_img_files.append(str(pred_img_path))
        # else:
        #     continue

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
    file1_path = get_versioned_filename(Path(MHD_RAW_UPLOAD_DIR) / file1.filename)
    file2_path = get_versioned_filename(Path(MHD_RAW_UPLOAD_DIR) / file2.filename)
    
    with open(file1_path, "wb") as f:
        f.write(await file1.read())

    with open(file2_path, "wb") as f:
        f.write(await file2.read())

    # Xử lý file với SimpleITK
    # result_files, pred_files = process_mhd_raw_files(file1_path, file2_path, file1.filename)
    result_files, pred_files = process_mhd_raw_files(file1_path, file2_path, file1_path.stem)
    image_urls = [f"http://localhost:8000/output_mhd_raw/{Path(file).name}" for file in result_files]
    pred_image_urls = [f"http://localhost:8000/predict_image_mhd_raw/{Path(file).name}" for file in pred_files]
    
    # file_id = file1.filename[:-4]  # Sử dụng tên file `.mhd` làm ID
    file_id = file1_path.stem  # Sử dụng tên file `.mhd` làm ID
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
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1))
    img_tensor = img_tensor.float().div(255)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        prediction = FRCNN_model([img_tensor])[0]
    prediction = apply_nms(prediction)
    score_threshold = 0.25

    # lọc các box, label và score với điều kiện score > 0.25
    filtered_indices = prediction["scores"] > score_threshold
    filtered_boxes = prediction["boxes"][filtered_indices]
    filtered_labels = prediction["labels"][filtered_indices]
    filtered_scores = prediction["scores"][filtered_indices]

    # kết quả sau khi lọc
    filtered_data = {
        "boxes": filtered_boxes,
        "labels": filtered_labels,
        "scores": filtered_scores
    }
    result = filtered_data["boxes"]
    if len(result) != 0:
        for box in result:
            x_min_frcnn, y_min_frcnn, x_max_frcnn, y_max_frcnn = box.tolist()
            cv2.rectangle(img, (int(x_min_frcnn), int(y_min_frcnn)), (int(x_max_frcnn), int(y_max_frcnn)), color=(0,0,255), thickness=2)
    output_path = f"{file.filename.rsplit('.', 1)[0]}.png"
    pred_img_path = PNG_PREDICT_SLICE_PNG_DIR / output_path
    cv2.imwrite(pred_img_path, img)
    predicted_image_url = f"http://localhost:8000/predict_image_png/{output_path}"
    upload_image_url = f"http://localhost:8000/uploads_png/{file.filename}"
        
    # box = YOLO_model(img)
    # if len(box[0].boxes.xyxy) != 0:
    #     x_min_yolo, y_min_yolo, x_max_yolo, y_max_yolo = box[0].boxes.xyxy[0].tolist()
    #     cv2.rectangle(img, (int(x_min_yolo), int(y_min_yolo)), (int(x_max_yolo), int(y_max_yolo)), color=(0,0,255), thickness=2)
    # output_path = f"{file.filename.rsplit('.', 1)[0]}.png"
    # pred_img_path = PNG_PREDICT_SLICE_PNG_DIR / output_path
    # cv2.imwrite(pred_img_path, img)
    # predicted_image_url = f"http://localhost:8000/predict_image_png/{output_path}"
    # upload_image_url = f"http://localhost:8000/uploads_png/{file.filename}"
    
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

@app.post("/upload_dcm/")
async def upload_folder(files: List[UploadFile] = File(...)):
    folder_name = files[0].filename.split('/')[0]
    upload_file_paths = []
    for file in files:
        file_path = Path(DCM_UPLOAD_DIR) / f"{folder_name}_{file.filename.split('/')[1]}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        upload_file_paths.append(file_path)
    
    result_files, pred_files= process_dcm_files(upload_file_paths, folder_name)
    image_urls = [f"http://localhost:8000/output_dcm/{Path(file).name}" for file in result_files]
    pred_image_urls = [f"http://localhost:8000/predict_image_dcm/{Path(file).name}" for file in pred_files]
    
    save_extraction_history(file.filename.split("/")[0], "dcm", image_urls, pred_image_urls)
    
    return {"files": image_urls, "pred_files": pred_image_urls}

@app.get("/output_dcm/{image_name}")
async def get_image(image_name: str):
    image_path = Path(DCM_OUTPUT_DIR) / image_name
    if image_path.exists():
        return FileResponse(image_path)
    return {"error": "Image not found"}

@app.get("/predict_image_dcm/{image_name}")
async def get_predict_image(image_name: str):
    image_path = Path(DCM_PREDICT_SLICE_PNG_DIR) / image_name
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
    if not record:
        raise HTTPException(status_code=404, detail="History record not found")
    history_entries = db.query(RecognitionHistory).filter(
        RecognitionHistory.file_name == record.file_name
    ).all()

    if len(history_entries) == 1:
        if record.file_type == "mhd/raw":
            for mhd_file in Path(MHD_RAW_UPLOAD_DIR).glob(f"{record.file_name}*.mhd"):
                mhd_file.unlink(missing_ok=True)
            for raw_file in Path(MHD_RAW_UPLOAD_DIR).glob(f"{record.file_name}*.raw"):
                raw_file.unlink(missing_ok=True)
            
            # Xóa các file ảnh trong thư mục output
            for output_file in Path(MHD_RAW_OUTPUT_DIR).glob(f"{record.file_name}_*.png"):
                output_file.unlink(missing_ok=True)

            # Tìm tất cả các file .jpg trong thư mục predict_image_dir
            for predicted_file in Path(MHD_RAW_PREDICT_SLICE_PNG_DIR).glob(f"{record.file_name}_*.png"):
                predicted_file.unlink(missing_ok=True)
                
        elif record.file_type == "png/jpg/jpeg":
            for predicted_file in Path(PNG_PREDICT_SLICE_PNG_DIR).glob(f"{record.file_name}*.png"):
                predicted_file.unlink(missing_ok=True)
                
        elif record.file_type == "dcm":
            for dcm_file in Path(DCM_UPLOAD_DIR).glob(f"{record.file_name}*.dcm"):
                dcm_file.unlink(missing_ok=True)
            
            # Xóa các file ảnh trong thư mục output
            for output_file in Path(DCM_OUTPUT_DIR).glob(f"{record.file_name}_*.png"):
                output_file.unlink(missing_ok=True)

            # Tìm tất cả các file .jpg trong thư mục predict_image_dir
            for predicted_file in Path(DCM_PREDICT_SLICE_PNG_DIR).glob(f"{record.file_name}_*.png"):
                predicted_file.unlink(missing_ok=True)

    # Xóa bản ghi trong cơ sở dữ liệu
    db.delete(record)
    db.commit()

    return {"message": "History record deleted successfully"}