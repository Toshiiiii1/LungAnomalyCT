from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
import SimpleITK as sitk
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS handle
origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Tạo thư mục tạm cho các file PNG
PNG_DIR = Path("png_images")
PNG_DIR.mkdir(exist_ok=True)

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

def convert_mhd_to_png(mhd_path):
    # Đọc ảnh y khoa từ file .mhd và .raw
    itk_image = sitk.ReadImage(str(mhd_path))
    # Chuyển đổi ảnh từ định dạng ITK sang array numpy
    image_array = sitk.GetArrayFromImage(itk_image)

    # Xóa tất cả các file ảnh cũ có định dạng .png nếu tồn tại trong thư mục PNG_DIR
    for file in PNG_DIR.glob("*.png"):
        file.unlink()

    # Lưu từng lát cắt của ảnh dưới dạng file .png
    png_files = []  # Danh sách để lưu đường dẫn các file .png đã lưu
    for i, slice_array in enumerate(image_array):
        img = normalize(slice_array)  # Chuẩn hóa lát cắt ảnh
        img_grey = img * 255  # Chuyển đổi ảnh chuẩn hóa sang ảnh xám (giá trị từ 0-255)
        img_rgb = np.stack((img_grey,) * 3, -1)  # Tạo ảnh RGB từ ảnh xám
        png_path = PNG_DIR / f"slice_{i}.png"  # Đường dẫn lưu file .png
        cv2.imwrite(png_path, img_rgb)  # Lưu ảnh dưới dạng file .png
        png_files.append(str(png_path))  # Thêm đường dẫn file .png vào danh sách

    return png_files  # Trả về danh sách đường dẫn các file .png đã lưu

# files = convert_mhd_to_png(r"D:\Code\Python Code\CT552\LUNA16\test\1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249.mhd")

@app.get("/convert/{filename}")
async def convert_and_get_images(filename: str):
    mhd_path = Path(r"D:\Code\Python Code\CT552\LUNA16\test") / filename
    if not mhd_path.exists():
        return {"error": "File not found"}

    # Chuyển đổi các ảnh từ .mhd sang .png và trả về danh sách ảnh
    png_files = convert_mhd_to_png(mhd_path)
    image_urls = [f"http://localhost:8000/images/{Path(file).name}" for file in png_files]
    return {"images": image_urls}

# Endpoint để truy cập ảnh .png
@app.get("/images/{image_name}")
async def get_image(image_name: str):
    image_path = Path(PNG_DIR) / image_name
    if image_path.exists():
        return FileResponse(image_path)
    return {"error": "Image not found"}
