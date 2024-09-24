from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import os
import torchvision
import matplotlib.pyplot as plt
import warnings
from PIL import Image, ImageDraw
from ultralytics import YOLO
import cv2
import numpy as np
import streamlit as st
import pandas as pd

warnings.filterwarnings("ignore")

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor):
        ann_file = os.path.join(img_folder, "_annotations.coco.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

if __name__ == "__main__":
    processor = DetrImageProcessor.from_pretrained("Toshiiiii1/detr-lung-ct5")
    test_dataset = CocoDetection(img_folder=r"D:\Code\Python Code\CT552\lung coco\test", processor=processor)
    cats = test_dataset.coco.cats
    id2label = {k: v["name"] for k, v in cats.items()}
    model = DetrForObjectDetection.from_pretrained("Toshiiiii1/detr-lung-ct5", id2label=id2label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    YOLO_model = YOLO(r"D:\Code\Python Code\CT552\runs_yolov9_3\train\weights\best.pt")
    
    uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        col1, col2, col3 = st.columns(3)
        
        img_path = os.path.join(r"D:\Code\Python Code\CT552\lungct.v1i.yolov9\test\images", uploaded_file.name)
        roi_id = uploaded_file.name.rsplit("_", 1)[0].replace("-", ".")
        df_roi = pd.read_csv("./ROI_coor2.csv")
        a = df_roi[df_roi["ID"] == f"{roi_id}"]
        x_min, x_max, y_min, y_max = a["x_min"].iloc[0], a["x_max"].iloc[0], a["y_min"].iloc[0], a["y_max"].iloc[0]
        img = cv2.imread(img_path)
        img1 = cv2.resize(img, (512, 512))
        cv2.rectangle(img1, (x_min, y_min), (x_max, y_max), color=(0,255,0), thickness=2)
        with col1:
            st.image(img1, caption='Ảnh đã chọn', use_column_width=True)
        
        img2 = cv2.resize(img, (1024, 1024))
        encoding = processor(images=img2, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        pixel_values = pixel_values.unsqueeze(0)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=None)
        
        postprocessed_outputs = processor.post_process_object_detection(outputs, target_sizes=[(1024, 1024)], threshold=0.5)
        results = postprocessed_outputs[0]
        if results["boxes"].tolist():
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                x_min, y_min, x_max, y_max = box.tolist()
                cv2.rectangle(img2, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(255,0,0), thickness=2)
        with col2:
                st.image(img2, caption="DETR", use_column_width=True)
            
        img3 = cv2.resize(img, (1024, 1024))
        box = YOLO_model(img3, max_det=1)
        x_min, y_min, x_max, y_max = box[0].boxes.xyxy[0].tolist()
        cv2.rectangle(img3, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(255,0,0), thickness=2)
        with col3:
            st.image(img3, caption="YOLO", use_column_width=True)