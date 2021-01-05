import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import nni

from dataset_model import CropDataset

class CropDataModule(pl.LightningDataModule):

    def setup(self, stage):
        # transforms
        transforms_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])

        transforms_valid = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
            
        # prepare transformations

        data = [] # contains [img, label]
        classes_ct = {}

        for idx, folder in enumerate(folders):
            for file in os.listdir(os.path.join(path, folder)): # all images
                img_path = os.path.join(path, folder, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))
                # get labels (can be defined by folder name)
                label = idx
                data.append([img, idx])
                # count images in classes
                if label not in classes_ct.keys():
                    classes_ct[label] = 0
                else:
                    classes_ct[label] +=1

        print(f'Classes: {classes}')
        print(f'Nr of images in each class: {classes_ct}\n')

        X = np.array([i[0] for i in data])
        y = np.array([i[1] for i in data])

        # define train and validation set
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)

        print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
        print(f'X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}\n')

        # save as dataset
        self.train_dataset = CropDataset(X_train, y_train, transform=transforms_train)
        self.valid_dataset = CropDataset(X_valid, y_valid, transform=transforms_valid)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=batch_size, num_workers=8)

class CropClassifierLightning(pl.LightningModule):
    def __init__(self, img_size, hidden_size, conv_size1, conv_size2, conv_size3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv_size1, 3) # input channel, output, kernel size
        self.conv2 = nn.Conv2d(conv_size1, conv_size2, 3)
        self.conv3 = nn.Conv2d(conv_size2, conv_size3, 3)

        # find size of following fc layer
        x = torch.randn(3, img_size, img_size).view(-1, 3, img_size, img_size) # first dim is batch size
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 5) # 5 classes

        self.lr = 0.001

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        if self._to_linear == None:
            self._to_linear = np.prod(x[0].shape)

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear) # flatten x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x, y = valid_batch  
        logits = self.forward(x)
        valid_loss = self.cross_entropy_loss(logits, y)
        self.log('valid_loss', valid_loss)
        nni.report_intermediate_result(float(valid_loss))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer


if __name__ == '__main__':

    # directories, where images are saved
    path = 'data/crop_images'
    folders = ['jute', 'maize', 'rice', 'sugarcane', 'wheat']

    img_size = 224
    batch_size = 32
    classes = {'jute': 0, 'maize': 1, 'rice': 2, 'sugarane': 3, 'wheat': 4}

    data_module = CropDataModule()

    logger = TensorBoardLogger('tb_logs', name='my_first_test')

    hparams = nni.get_next_parameter()
    hidden_size=hparams['hidden_size']
    conv_size1, conv_size2, conv_size3 = hparams['conv_size1'], hparams['conv_size2'], hparams['conv_size3']
    # train
    model = CropClassifierLightning(img_size=img_size, hidden_size=hidden_size,
                                    conv_size1=conv_size1, conv_size2=conv_size2, conv_size3=conv_size3)

    early_stop_callback = EarlyStopping(monitor='valid_loss',
                                        min_delta=0.00,
                                        patience=5,
                                        verbose=False,
                                        mode='max')
    trainer = pl.Trainer(callbacks=[early_stop_callback], max_epochs=100, logger=logger)

    trainer.fit(model, data_module)
    
    torch.save(model.state_dict(), 'model_crop_lightning.pt')
