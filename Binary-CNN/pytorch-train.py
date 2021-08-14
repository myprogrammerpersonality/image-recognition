# imports
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define a CNN
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3, padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        
        self.fc1 = nn.Linear(3*3*64,10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = self.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, annotations_file_name='label.csv', transform=None, target_transform=None):
        self.img_labels = pd.read_csv(os.path.join(img_dir, annotations_file_name))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    
def main(args):

    # Data augmentation and normalization for training
    # Just normalization for validation
    input_size = 224

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),}
    train_data = CustomImageDataset(args.train, annotations_file_name='label.csv', transform=data_transforms['train'], target_transform=None)
    val_data = CustomImageDataset(args.val, annotations_file_name='label.csv', transform=data_transforms['val'], target_transform=None)
    
    
    # make dataloader objects
    batch_size = args.batch_size

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classes =   {0:'cat',
                 1:'dog'}
    
    # define CNN object
    
    net = Net().to(device)
    # should be changed for multiclass to cross entropy

    criterion = nn.BCELoss()
    optimizer = optim.Adam(params = net.parameters(), lr=args.learning_rate)
    
    loss_history_dict = {'train':[],
                     'val':[]}

    dataloaders = {'train':train_dataloader,
                   'val':val_dataloader}

    for epoch in range(args.epochs):  # loop over the dataset multiple times
      
        for phase in ['train', 'val']:
                    if phase == 'train':
                        net.train()  # Set model to training mode
                    else:
                        net.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    for i, data in enumerate(train_dataloader, 0):
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data[0].to(device), data[1].to(device)
                    
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        outputs = net(inputs.float())
                        loss = criterion(outputs, labels.float().unsqueeze(1))
                    
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        # print statistics
                        running_loss += loss.item()

                        if i % 500 == 499:   # print every 500 mini-batches
                            avg_loss = running_loss / 500
                            loss_history_dict[phase].append(avg_loss)
                            print(f'[{epoch+1}, {i+1}] {phase} loss: {avg_loss}')
                            running_loss = 0.0
    
    with open(os.path.join(args.model_dir, 'cat_dog_net.pth'), 'wb') as f:
        torch.save(net.state_dict(), f)
    
if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))

    args, _ = parser.parse_known_args()
    
    main(args)