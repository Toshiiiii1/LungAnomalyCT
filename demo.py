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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

warnings.filterwarnings("ignore")

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

id2label = {0: 'lung_ct_2', 1: 'nodule'}

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor):
        ann_file = os.path.join(img_folder, "_annotations.coco.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)

        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
    
def apply_nms(orig_prediction, iou_thresh=0.3):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

def plot_results(pil_img, scores, labels, boxes, model="yolo"):
    # Tạo figure và axes cho matplotlib
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(pil_img)

    # Lặp qua các bounding box, nhãn và điểm số để vẽ
    colors = COLORS * 100  # Nhân đôi màu để có đủ màu sắc nếu cần
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        # Vẽ khung hình chữ nhật
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, edgecolor=c, linewidth=3))
        
        # Thêm nhãn và điểm số
        if model == "yolo":
            text = f'{id2label[1]}: {score:0.2f}'
        else:
            text = f'{id2label[label]}: {score:0.2f}'
        # ax.text(xmin, ymin, text, fontsize=15, color='black', bbox=dict(facecolor='yellow', alpha=0.5))
    
    # Tắt hiển thị các trục
    ax.axis('off')
    
    # Hiển thị hình ảnh trên Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    # load mô hình DETR
    processor = DetrImageProcessor.from_pretrained(r"D:\Code\Python Code\CT552\DETR results\runs_detr_R50_DC5\checkpoint-15150")
    test_dataset = CocoDetection(img_folder=r"D:\Code\Python Code\CT552\COCO format\lung_ct_version_n_512.v2i.coco\test", processor=processor)
    cats = test_dataset.coco.cats
    id2label = {k: v["name"] for k, v in cats.items()}
    model = DetrForObjectDetection.from_pretrained(r"D:\Code\Python Code\CT552\DETR results\runs_detr_R50_DC5\checkpoint-15150", id2label=id2label, ignore_mismatched_sizes=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # load mô hình YOLO
    YOLO_model = YOLO(r"D:\Code\Python Code\CT552\YOLO results\runs_yolov8x_512_new\train\weights\best.pt")
    
    # load mô hình Faster RCNN
    FRCNN_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = FRCNN_model.roi_heads.box_predictor.cls_score.in_features
    FRCNN_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2) 
    checkpoint = torch.load(r"D:\Code\Python Code\CT552\FRCNN result\frcnn_model_3_512.pth", map_location=torch.device('cpu'))
    FRCNN_model.load_state_dict(checkpoint['model_state_dict'])
    FRCNN_model.to(device)
    FRCNN_model.eval()
    
    # chọn ảnh
    uploaded_file = st.file_uploader("Chọn ảnh...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        col1, col2, col3 = st.columns(3)
        
        # lấy thông tin về tọa độ bounding box từ file csv, hiển thị bouding box lên ảnh
        img_path = os.path.join(r"D:\Code\Python Code\CT552\YOLO format\lung_ct_version_n_512.v2i.yolov8\test\images", uploaded_file.name)
        roi_id = uploaded_file.name.rsplit("_", 1)[0].replace("-", ".")
        df_roi = pd.read_csv("./ROI_coor4.csv")
        a = df_roi[df_roi["ID"] == f"{roi_id}"]
        x_min, x_max, y_min, y_max = a["x_min"].iloc[0], a["x_max"].iloc[0], a["y_min"].iloc[0], a["y_max"].iloc[0]
        img = cv2.imread(img_path)
        img1 = cv2.resize(img, (512, 512))
        cv2.rectangle(img1, (x_min, y_min), (x_max, y_max), color=(0,255,0), thickness=2)
        st.image(img1, caption='Ảnh đã chọn', use_column_width=False)
        
        # sử dụng mô hình DETR để xác định bouding box
        img2 = cv2.resize(img, (512, 512))
        encoding = processor(images=img2, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        pixel_values = pixel_values.unsqueeze(0)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=None)
        postprocessed_outputs = processor.post_process_object_detection(outputs, target_sizes=[(512, 512)], threshold=0.5)
        results = postprocessed_outputs[0]
        # st.write(results)
        image = Image.open(img_path)
        # plot_results(image, results['scores'], results['labels'], results['boxes'], model)
        # if results["boxes"].tolist():
        #     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        #         x_min_detr, y_min_detr, x_max_detr, y_max_detr = box.tolist()
        #         cv2.rectangle(img2, (int(x_min_detr), int(y_min_detr)), (int(x_max_detr), int(y_max_detr)), color=(255,0,0), thickness=2)
        # cv2.rectangle(img2, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0,255,0), thickness=2)
        with col1:
            st.caption("DETR")
            plot_results(image, results['scores'], results['labels'], results['boxes'])
            # st.image(img2, caption="DETR", use_column_width=True)
        
        # sử dụng mô hình YOLO để xác định bouding box    
        img3 = cv2.resize(img, (512, 512))
        box = YOLO_model(img3)
        # st.write(box[0].boxes)
        # x_min_yolo, y_min_yolo, x_max_yolo, y_max_yolo = box[0].boxes.xyxy[0].tolist()
        # cv2.rectangle(img3, (int(x_min_yolo), int(y_min_yolo)), (int(x_max_yolo), int(y_max_yolo)), color=(255,0,0), thickness=2)
        # cv2.rectangle(img3, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0,255,0), thickness=2)
        with col2:
            # st.image(img3, caption="YOLO", use_column_width=True)
            st.caption("YOLO")
            plot_results(image, box[0].boxes.conf, box[0].boxes.cls, box[0].boxes.xyxy)
        
        # sử dụng mô hình Faster RCNN để xác định bouding box    
        img4 = torch.from_numpy(img3.transpose(2, 0, 1))
        img4 = img4.float().div(255)
        with torch.no_grad():
            prediction = FRCNN_model([img4])[0]
        prediction = apply_nms(prediction)
        # st.write(prediction)
        # thiết lập ngưỡng
        score_threshold = 0.1

        # lọc các box, label và score với điều kiện score > 0.1
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
        # for box in result:
        #     x_min_frcnn, y_min_frcnn, x_max_frcnn, y_max_frcnn = box.tolist()
        #     cv2.rectangle(img3, (int(x_min_frcnn), int(y_min_frcnn)), (int(x_max_frcnn), int(y_max_frcnn)), color=(255,0,0), thickness=2)
        # cv2.rectangle(img3, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0,255,0), thickness=2)
        with col3:
            st.caption("Faster RCNN")
            plot_results(image, filtered_data['scores'], filtered_data['labels'], filtered_data['boxes'])
                # st.image(img3, caption="Faster RCNN", use_column_width=True)