import os
import torch
import utils
import argparse

from engine import train_one_epoch, evaluate
from data import LungImagesDataset
from model import CustomFasterRCNN

def parse_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--weight", type=str, default="", help="Pre-trained model weight")
    parser.add_argument("--train-set", type=str, default="", help="Train data in VOC format")
    parser.add_argument("--val-set", type=str, default="", help="Validation data in VOC format")
    
    opt = parser.parse_args()
    
    return opt
        
def main():
    opt = parse_opt()
    
    print("Preparing data...")
    # use our dataset and defined transformations
    train_dataset = LungImagesDataset(opt.train_set, 512, 512, is_train=True)
    val_dataset = LungImagesDataset(opt.val_set, 512, 512, is_train=False)

    # define training and validation data loaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    
    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    num_epochs = 1

    print("Loading model...")
    # get the model using our helper function
    if opt.weight:
        model = CustomFasterRCNN(num_classes=num_classes, checkpoint_path=opt.weight).get_model()
    else:
        model = CustomFasterRCNN(num_classes=num_classes).get_model()

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
        # evaluate on the validate dataset
        evaluate(model, val_data_loader, device=device)
    
    ckpt_file_name = f"/kaggle/working/frcnn_model_1.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_file_name)

if __name__ == "__main__":
    # TODO: adjust RPN anchor size
    
    main()