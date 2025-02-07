# Created on Feb 4th, 2025
# Auth. Uriel Garcilazo Cruz

from VAE import VAE
import os 
from os.path import join as jn 
import torch 
from PIL import Image 
import numpy as np 
from torch.utils.data import Dataset
from latent_grad_vis import latent_grad_vis
from constants import *
from knn import analyze_latent_knn

device = "cuda" if torch.cuda.is_available() else "cpu"

# DD. MODEL
# model = Vae()
# purp. Load the VAE model
model = VAE()
model.load_state_dict(torch.load(PATH_MODEL))
model = model.to(device)

# FOLDER PATH
# path = "str"
# purp. Load the validation dataset
# INVARIANT:
# folder structure follows:
# --- path
#   |--- phenotype_1
#   |--- phenotype_2

path = "../03_subset_stnd"

def reparameterization(mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    



def get_encodings(np_array_images,model):
    x_dataset_tensor = torch.tensor(np_array_images).to(device)
    # print(x_dataset_tensor.shape)
    mean,log_var = model.encode(x_dataset_tensor)
    # print(mean.shape,log_var.shape)
    # first element is the mean from the encoding
    # second element is the logvar from the encoding
    # mean and logvar enter reparameterization to get latent vector z
    z = reparameterization(mean, torch.exp(0.5 * log_var))
    return (z)
    
def encode_images(path,model,res=128):
    # for every folder in the path
    #   for every image in the folder
    #       load the image
    #       preprocess the image
    #       encode the image
    #       save the encoding
    final_encodings = []
    for folder in os.listdir(path):
        images_folder = [] 
        for image in os.listdir(jn(path,folder)):
            img_path = jn(path, folder,image)
            # Open image using PIL
            img = Image.open(img_path)
            # Resize image to target resolution
            img_resized = img.resize((res, res))
            # Convert to numpy array
            img_array = np.array(img_resized)/255.
            img_array = img_array.astype(np.float32)
            
            images_folder.append(img_array)
            # img = img.unsqueeze(0)
        x_set_images = np.stack(images_folder)
        # Permute the dimensions from (batch, height, width, channels)
        # to (batch, channels, height, width)
        x_set_images = np.transpose(x_set_images, (0, 3, 1, 2))
        # torch.save(encoding,jn(path,folder,image+".enc"))
        # print("encodings for ",folder)
        final_encodings.append(get_encodings(x_set_images,model))
    return final_encodings

latent_space_sets = encode_images(path, model)
# print(latent_space_sets) #To see the shape of the encoding


# reshape the tensor to a list of lists
# reshape list of lists to a list of floats
phenotypes = [tensor.cpu().detach().numpy().tolist() for tensor in latent_space_sets]
phenotypes = [[col for col in row] for row in phenotypes]

# To generate the visualization 

# input phenotype 1 and phenotype 2, VAE model and number of images to generate
# generate visualization by sampling range within the latent space set by the two phenotypes
latent_grad_vis(phenotypes[0],phenotypes[1],model,n=10)

# Calculate knn values for each phenotype
for phenotype in phenotypes:
    distances, indices = analyze_latent_knn(phenotype, k=4)
    print(distances, indices) # To see the distances and indices of the knn