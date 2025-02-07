# Created on Jan 30th, 2025
# Auth. Uriel Garcilazo Cruz 
# Generate a latent space via variational autoencoder (VAE).
# The VAE uses the weights and biases from the VGG model.

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset
from os.path import join as jn
import os
from VAE import VAE
from PIL import Image 
from constants import *
# Algorithm:
# - Load the dataset at resolution 128x128 from numpy
# - Generate Dataset meta-object for DataLoader






# create train and test dataloaders
batch_size = 30


# # FD. create_subdataset()
# # purp. generate a subdataset of the quickdraw dataset, which contains over 4 million datapoints (too much for commercial small GPUs)
# def create_subdataset(images, labels, sample_size=100000):
#     assert images.shape[0] == labels.shape[0], "The number of images and labels must match"

#     # Get the total number of data points
#     total_data_points = images.shape[0]

#     # Randomly select sample_size indices
#     random_indices = np.random.choice(total_data_points, size=sample_size, replace=False)

#     # Extract the sampled images and labels
#     sampled_images = images[random_indices]
#     sampled_labels = labels[random_indices]

#     return sampled_images, sampled_labels

# # FD. loadQuickDraw()
# # purp. load the quickdraw dataset, saved as a set of .npy files
# # Find the folder with the numpy datapoints
#     # for every folder:
#     #   read the data as np, append it to the final array
#     #   create a label (int) representing the index of the folder found, and propagate for the shape[0] of the data in that folder
#     #   add the label name to the list of indexes label_names
    
# def loadQuickDraw(path):
#     np_dataset = None
#     np_labels = None
#     files = os.listdir(path)
#     for i,file in enumerate(files):
#         # Trim and add the category name to label names
#         labelName = file.split("_")[-1]
#         labelName = labelName.replace(".npy","")
#         labelName = labelName.replace(" ","")
#         labelName = labelName.strip()
#         label_names.append(labelName)
        
#         filepath = jn(path,file)
#         print(filepath)
#         file = np.load(filepath)
#         np_label = np.full(file.shape[0], i)
#         # print(np_label.shape)
#         if i == 0:
#             np_dataset = file
#             np_labels = np_label
#         else:
#             np_dataset = np.vstack((np_dataset, file))
#             np_labels = np.concatenate((np_labels, np_label))
#     return np_dataset/255.,np_labels
        


# # FD. loadDataset()
# Signature: str int -> np.array
# purpose: load the dataset from the path
def loadDataset(path, res=128):
    files = [i for i in os.listdir(path) if i.endswith(".png")]
    images = []
    # Load and process each image
    for img_file in files:
        img_path = jn(path, img_file)
        # Open image using PIL
        img = Image.open(img_path)
        # Resize image to target resolution
        img_resized = img.resize((res, res))
        # Convert to numpy array
        img_array = np.array(img_resized)/255.
        img_array = img_array.astype(np.float32)
        
        images.append(img_array)
    # stack the images into a single np array
    x_train = np.stack(images)
    # Permute the dimensions from (batch, height, width, channels)
    # to (batch, channels, height, width)
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    return x_train
    


x_train = loadDataset("../02_stnd_Dataset",128)

# x_train = x_train.reshape(-1,1,28,28).astype(np.float32)





# DD. TRAIN_DATASET
# train_dataset = Dataset(x_train, y_train)
# interp. preparing the data to enter DataLoader as a member of the class Dataset
# A child class from Dataset is created
class Dataset(Dataset):
    def __init__(self,x_train,y_train=None):
        self.x = x_train
        # self.y = y_train 
    def __getitem__(self,index):
        # return self.x[index],self.y[index]
        return self.x[index]
    def __len__(self):
        return len(self.x)

train_dataset = Dataset(x_train)


# #Place dataset into the DataLoader that controls the batch sizes
train_load = torch.utils.data.DataLoader(dataset = train_dataset, 
                                         batch_size = batch_size,
                                         shuffle = True)




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

def train(model, optimizer, epochs, device, x_dim=128*128):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(train_load):
            # print(x.shape)
            # x = x.view(x.shape[0], x_dim).to(device)
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
    return overall_loss

# for param in model.features.parameters():
#     param.require_grad = False
for name, param in model.named_parameters():
    if "encoder" in name:
        print(name, param.requires_grad)
        # param.requires_grad = False
        # print(name, param.requires_grad)
        # param.requires_grad = False
        
train(model, optimizer, epochs=350, device=device)
torch.save(model.state_dict(), "output_VAE_VGG_Feb6_10z.pth")
