import SimpleITK as sitk
import cv2
import os
import torch
import torchvision
import numpy as np
import argparse

from model import CustomFasterRCNN
from tqdm import tqdm
from pathlib import Path

def apply_nms(orig_prediction, iou_thresh=0.3):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

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

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--weight", type=str, default="", help="Pre-trained model weight")
    parser.add_argument("--mhd", type=str, default="", help="MHD file")
    
    opt = parser.parse_args()
    
    return opt

def main():
    opt = parse_opt()
    # Đọc ảnh y khoa từ file .mhd và .raw
    itk_image = sitk.ReadImage(str(opt.mhd))
    # Chuyển đổi ảnh từ định dạng ITK sang array numpy
    image_array = sitk.GetArrayFromImage(itk_image)
    
    id = os.path.basename(opt.mhd).rsplit('.', 1)[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FRCNN_model = CustomFasterRCNN(num_classes=2, checkpoint_path=opt.weight, is_train=False).get_model()
    
    output_folder = Path(f"data/predicted/{id}")
    os.makedirs(output_folder, exist_ok=True)

    for i, slice_array in tqdm(enumerate(image_array)):
        img = normalize(slice_array, "mhd/raw")  # Chuẩn hóa lát cắt ảnh
        img_grey = img * 255  # Chuyển đổi ảnh chuẩn hóa sang ảnh xám (giá trị từ 0-255)
        img_rgb = np.stack((img_grey,) * 3, -1)  # Tạo ảnh RGB từ ảnh xám
        
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
            pred_img_path = output_folder / f"{id}_slice_{i}_predicted.png"
            cv2.imwrite(pred_img_path, img_rgb)
        else:
            continue

if __name__ == "__main__":
    main()