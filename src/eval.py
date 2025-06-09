import argparse
import torch
import utils

from engine import evaluate
from model import CustomFasterRCNN
from data import LungImagesDataset

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--weight", type=str, default="", help="Pre-trained model weight")
    parser.add_argument("--data", type=str, default="", help="Data in VOC format")
    
    opt = parser.parse_args()
    
    return opt

def main():
    opt = parse_opt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CustomFasterRCNN(num_classes=2, checkpoint_path=opt.weight, is_train=False).get_model()
    
    test_dataset = LungImagesDataset(opt.data, 512, 512, is_train=False)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    
    evaluate(model, test_data_loader, device=device)

if __name__ == "__main__":
    main()