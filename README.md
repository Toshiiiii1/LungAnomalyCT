# Abnormal detection on lung CT scans

This is a tool automatically detects locations that are believed to be lung nodule from lung CT scan slices. Click  [here](./docs/README.md) to see more details.

## Features
- Automatically detect and localize locations contain lung nodules.
- Has API can be intergrated into other application.
- Support `.dcm` file format is used in hospitals or normal image (png/jpg).

## Quickstart

### Run demo with web interface

**GPU is recommended to speed up detection**

1. Installation

```bash
# Clone the repo
git clone https://github.com/Toshiiiii1/Abnormal_religions_detection_on_lung_CT_image.git

# Start virtual eviroment
python -m venv venv
source venv/Scripts/activate

# Check venv activate
which python

# Install the required Python libraries
pip install -r requirements.txt
```

2. Start backend (FastAPI)
```bash
# Move to demo folder
cd demo/

# Initialize database (store detection history)
python models.py

# Move to server folder and start backend
cd server/
uvicorn main:app --port 8000
```

3. Start frontend (ReactJS)
```bash
# Move to frontend folder
cd demo/frontend/

# Install required libraries
npm install

# Start frontend
npm run dev
```

### [Video demo](https://drive.google.com/file/d/1vbjmvar-hP2iIdj4G4l4DCJAoNS9rWrj/view?usp=drive_link)

## Technical details
- Detection models: YOLOv8, DETR ResNet50 and Faster R-CNN ResNet50.
- Backend: FastAPI.
- UI: ReactJS.
- Other: HuggingFace, Pytorch, Numpy, Ultralytics.

## Acknowledgments
- [LUNA16](https://luna16.grand-challenge.org/) for provide the public lung CT dataset.
- [HuggingFace](https://huggingface.co/docs/transformers/model_doc/detr), [Pytorch](https://pytorch.org/vision/main/models/faster_rcnn.html) and [Ultralytics](https://docs.ultralytics.com/vi/models/yolov8/) for the amazing detection models.
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent framework to build APIs.
- [ReactJS](https://react.dev/) for the amazing framework.
- [SimpleITK](https://simpleitk.org/), [Pydicom](https://pydicom.github.io/) for the amazing library to process medical images.