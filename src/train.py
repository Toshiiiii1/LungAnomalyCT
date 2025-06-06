import os
import urllib.request
# List of URLs to download
urls = [
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py"
]

# Download each file
for url in urls:
    # Extract the filename from the URL
    filename = os.path.basename(url)
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

import torch
import torchvision
import cv2
import numpy as np
import albumentations as A
import utils

from torch.utils.data import DataLoader, Dataset
from xml.etree import ElementTree as et
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from albumentations.pytorch.transforms import ToTensorV2
from engine import train_one_epoch, evaluate
    


class LungImagesDataset(torch.utils.data.Dataset):

    def __init__(self, files_dir, width, height, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width
        
        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = [image for image in sorted(os.listdir(files_dir))
                        if image[-4:]=='.jpg']
        
        
        # classes: 0 index is reserved for background
        self.classes = ["background", 'nodule']

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        # reading the images and converting them to correct size and color    
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0
        
        # annotation file
        annot_filename = img_name[:-4] + '.xml'
        annot_file_path = os.path.join(self.files_dir, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]
        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            
            # bounding box
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            
            xmin_corr = (xmin/wt)*self.width
            xmax_corr = (xmax/wt)*self.width
            ymin_corr = (ymin/ht)*self.height
            ymax_corr = (ymax/ht)*self.height
            
            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
        
        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        # labels = torch.as_tensor(labels, dtype=torch.int64)
        labels = torch.ones((len(labels),), dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        # image_id = torch.tensor([idx])
        target["image_id"] = idx


        if self.transforms:
            
            # sample = self.transforms(image = img_res,
            #                          bboxes = target['boxes'],
            #                          labels = labels)

            sample = self.transforms(image = img_res)
            
            img_res = sample['image']
            # target['boxes'] = torch.Tensor(sample['bboxes'])
            
            
            
        return img_res, target

    def __len__(self):
        return len(self.imgs)
    
def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

def get_transform(train):
    if train:
        return A.Compose([
                            A.HorizontalFlip(0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0) 
                        ])
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ])
        
def main(train_dir, val_dir, test_dir):
    print("Preparing data...")
    # use our dataset and defined transformations
    train_dataset = LungImagesDataset(train_dir, 512, 512, transforms= get_transform(train=True))
    val_dataset = LungImagesDataset(val_dir, 512, 512, transforms= get_transform(train=False))
    test_dataset = LungImagesDataset(test_dir, 512, 512, transforms= get_transform(train=False))

    # define training and validation data loaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    
    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    num_epochs = 1

    print("Loading model...")
    # get the model using our helper function
    model = get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # training for 50 epochs
    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=302)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, val_data_loader, device=device)
        
    evaluate(model, test_data_loader, device=device)
    
    ckpt_file_name = f"/kaggle/working/frcnn_model_1.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_file_name)

if __name__ == "__main__":
    train_dir = '/kaggle/input/lung-ct-version-n-512/lung_ct_version_n_512.v2i.voc/train'
    val_dir = '/kaggle/input/lung-ct-version-n-512/lung_ct_version_n_512.v2i.voc/valid'
    test_dir = '/kaggle/input/lung-ct-version-n-512/lung_ct_version_n_512.v2i.voc/test'
    
    main(train_dir, val_dir, test_dir)