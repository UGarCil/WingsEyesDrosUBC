# File created on Feb 6th, 2025
# Auth. Uriel Garcilazo Cruz
'''
The following algorithm is embedded in this script:

Given two sets of values, representing latent space representations of a groupd of images,
we will create a gradient visualization by generating n images,
where the value fed into the encoder is given by the position n_i in a linear space from l_1 to l_2

- Takes two sets of image values, represented as a list of latent space  
- Takes the average for each list, returning two real numbers l_1 and l_2
- Creates a gradient visualization by generating n images, where the value fed into the encoder is given by the position n_i in a linear space from l_1 to l_2

This will enable us to visualize the sense that our model developed on the data.  

The code is fed into 

'''


import os 
from os.path import join as jn 
import numpy as np 
from constants import *
from matplotlib import pyplot as plt




def latent_grad_vis(list1,list2,model,n=10):
    def generate_images_from_modelVAE(z):
        # determine if z is a list or a single value and add an extra dimension if needed
        if isinstance(z, np.float64):
        # if z.shape[0] == 1:
            z_sample = torch.tensor([[z]], dtype=torch.float).to(device)
        else:
            z_sample = torch.tensor([z], dtype=torch.float).to(device)
        x_decoded = model.decode(z_sample)
        image = x_decoded.detach().cpu().squeeze(0) # reshape vector to 2d array
        # If RGB, rearrange channels to be last
        if image.shape[0] == 3:  
            image = image.permute(1, 2, 0)  # Shape: [128, 128, 3]
        return image

    # LLM generated on Feb 6th, 2025 by Claude v.3.5 Sonnet
    def visualize_images(images, digit_size=128):
        """Display a row of images from a list of tensors with shape [128,128,3]"""
        fig = plt.figure(figsize=(20, 4))
        
        for i, img in enumerate(images):
            ax = fig.add_subplot(1, len(images), i+1)
            # Convert tensor to numpy if needed
            if torch.is_tensor(img):
                img = img.cpu().numpy()
            ax.imshow(img)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        
    # list1 and list2 are lists of latent space representations
    # model is the model used to encode the images
    # n is the number of images to generate
    # returns a list of n images, each image is a numpy array
    # representing the image generated from the latent space representation
    
    # get the average of the latent space representations
    if Z_DIM == 1:
        l1 = sum([i[0] for i in list1])/len(list1)
        l2 = sum([i[0] for i in list2])/len(list2)
    else:
        l1 = np.mean(np.array(list1), axis=0)  # handles any n-dimensional input
        l2 = np.mean(np.array(list2), axis=0)  # handles any n-dimensional input
    
    # create a linear space from l1 to l2
    space = np.linspace(l1,l2,n)
    # create a list to store the images
    images = []
    
    # for each value in the linear space
    for s in space:
        # create a latent space representation
        # z = torch.tensor(s).to(device)
        # # decode the image
        # x = model.decode(z)
        image = generate_images_from_modelVAE(s)
        # append the image to the list
        images.append(image)
    visualize_images(images)
    # return images