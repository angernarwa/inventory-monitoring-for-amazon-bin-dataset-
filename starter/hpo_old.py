import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import EfficientNet_B4_Weights
import time
import logging
import sys
import os

import argparse

# to resolve 'OSError: image file is truncated' error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

 # https://discuss.pytorch.org/t/rmse-loss-function/16540/2
def RMSELoss(y_pred,y):
    return torch.sqrt(torch.mean((y_pred-y)**2))

def test(model, test_loader, criterion, device):
    logger.info("Testing started.")
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    rmse = RMSELoss(preds, labels.float())

    logger.info("Testing Loss: {:.2f}, Testing Accuracy: {:.2f}%, RSME: {}".format(
        total_loss,
        100.0 * total_acc,
        rmse
    ))
    logger.info("Testing completed.")

def train(model, train_loader, validation_loader, criterion, optimizer, device, epochs):
    logger.info("Training started.")
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(1, epochs + 1):
        for phase in ['train', 'valid']:
            logging.info(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)

                accuracy = running_corrects/running_samples
                rmse = RMSELoss(preds, labels.float())
                logger.info("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%) RMSE: {} Time: {}".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                            rmse,
                            time.asctime() 
                        )
                    )

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            break
    logger.info("Training completed.")    
    return model
    
def net(num_classes: int):
    model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False   

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.BatchNorm1d(num_features),
        nn.Linear(num_features, 512, bias=False),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.1, inplace=True),
        nn.Linear(512, num_classes, bias=False))
    return model

def create_data_loaders(data_dir: str, batch_size: int):
    transformers = {
                    "training": transforms.Compose([
                        transforms.RandomHorizontalFlip(p= 0.5),
                        transforms.Resize((224,224)),
                        transforms.RandomGrayscale(p= 0.5),
                        transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)]),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                    ]),
                    "testing": transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                    ]),
                    "validating": transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                    ])
                }

    data_train = datasets.ImageFolder(data_dir + "/train", transform = transformers["training"])
    train_loader = torch.utils.data.DataLoader(data_train, batch_size, shuffle = True)

    data_validation = datasets.ImageFolder(data_dir + "/valid", transform = transformers["validating"])
    validation_loader = torch.utils.data.DataLoader(data_validation, batch_size, shuffle = True)

    data_test = datasets.ImageFolder(data_dir + "/test", transform = transformers["testing"])
    test_loader = torch.utils.data.DataLoader(data_test, batch_size, shuffle = True)

    return train_loader, validation_loader, test_loader

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on Device {device}")

    model = net(args.num_classes)
    model = model.to(device)
    
    loss_criterion = nn.CrossEntropyLoss() #this loss function performs better for a classification problem like this, even if we are measuring RSME too
    optimizer = optim.AdamW(model.classifier.parameters(), lr = args.lr)
    
    train_loader, validation_loader, test_loader = create_data_loaders(args.data_path , args.batch_size)
    
    model = train(model, train_loader, validation_loader, loss_criterion, optimizer, device, args.epochs)
    
    test(model, test_loader, loss_criterion, device)
    
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
        logger.info("Model weights saved.")

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type = int ,
        default = 128, 
        metavar = "N",
        help = "input batch size for training (default : 128)"
    )
    parser.add_argument(
        "--epochs",
        type = int ,
        default = 11, 
        metavar = "N",
        help = "number of epochs to train (default : 11)"
    )
    parser.add_argument(
        "--lr",
        type = float ,
        default = 0.001, 
        metavar = "LR",
        help = "learning rate (default : 0.001)"
    )
    parser.add_argument("--num-classes", type = int, default = 5,  help = "number of items in a bin [1-5]") #for allowing training with the full dataset if needed
    
    parser.add_argument("--data-path", type = str, default = os.environ["SM_CHANNEL_TRAINING"], help = "path to a valid s3 uri bucket where the images datasets can be found") 
    parser.add_argument("--model-dir", type = str, default = os.environ["SM_MODEL_DIR"], help = "path to the output for the model") 
    
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout)) 
    main(args)
