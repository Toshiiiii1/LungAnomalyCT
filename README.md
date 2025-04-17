# Phát hiện nốt phổi trên ảnh CT phổi

Tool tự động phát hiện những vị trí được cho là nốt phổi từ các lát cắt của ảnh CT phổi. Chi tiết đọc tại [đây](./docs/README.md)

## Tính năng chính
- Phát hiện và khoang vùng những vị trí được cho là nốt phổi.
- API có thể tích hợp vào các ứng dụng khác.
- Hỗ trợ định dạng file .dcm được dùng trong các bệnh viện.

## Quickstart

### Chạy demo với giao diện web

**Khuyến khích sử dụng GPU để tăng tốc độ phát hiện (hơn 10 phút đối với CPU)**

1. Cài đặt

```bash
# Clone repo
git clone https://github.com/Toshiiiii1/Abnormal_religions_detection_on_lung_CT_image.git

# Tải các thư viện Python cần thiết
pip install -r requirements.txt
```

2. Chạy backend (FastAPI)
```bash
# Di chuyển đến thư mục demo
cd demo/

# Khởi tạo database (lưu lịch sử phát hiện)
python models.py

# Di chuyển đến thư mục server và chạy backend
cd server/
uvicorn main:app --port 8000
```

3. Chạy frontend (ReactJS)
```bash
# Di chuyển đến thư mục frontend
cd demo/frontend/

# Tải các thư viện cần thiết
npm install

# Chạy frontend
npm run dev
```

### [Video demo](https://drive.google.com/file/d/1vbjmvar-hP2iIdj4G4l4DCJAoNS9rWrj/view?usp=drive_link)

## Công nghệ sử dụng
- Mô hình phát hiện đối tượng: YOLOv8, DETR ResNet50 và Faster R-CNN ResNet50.
- Backend: FastAPI.
- UI: ReactJS.