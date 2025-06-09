import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class CustomFasterRCNN:
    def __init__(self, num_classes, checkpoint_path=None, is_train=True):
        self.num_classes = num_classes
        self.checkpoint = checkpoint_path
        self.is_train = is_train
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
    def get_model(self):
        if self.checkpoint:
            checkpoint = torch.load(self.checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        self.model.to(self.device)    
        self.model.eval() if self.is_train == False else self.model.train()
            
        return self.model