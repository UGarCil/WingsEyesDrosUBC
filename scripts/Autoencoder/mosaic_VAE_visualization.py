# This algorithm is designed to create a visualization of a Variational Encoder, optimized for pytorch.
# To run properly, it relies on the latent dimensions set to 2 to create a 2D projection in the x and y axes

# If there are issues with the execution of the algorithm, please 
# determine the number of dimensions in the latent space representation of the VAE.
# This information is encoded in the constant Z_DIM of the constants.py file.

# MODULES
import matplotlib.pyplot as plt
from VAE_validate import *


# DATA DEFINITIONS

def generate_images_from_modelVAE(z1, z2):
    z_sample = torch.tensor([[z1, z2]], dtype=torch.float).to(device)
    x_decoded = model.decode(z_sample)
    image = x_decoded.detach().cpu().squeeze(0) # reshape vector to 2d array
    # If RGB, rearrange channels to be last
    if image.shape[0] == 3:  
        image = image.permute(1, 2, 0)  # Shape: [128, 128, 3]
    return image

#img1: mean0, var1 / img2: mean1, var0
# mean, var = (0.0, 1.0)
# img = generate_images_from_modelVAE(mean, var)
# plt.title(f'[{mean},{var}]')
# plt.imshow(img)
# plt.axis('off')
# plt.show()



# generate_images_from_modelVAE(1.0, 0.0)
# FD. plot_latent_space()
# purp. Plot the latent space of the VAE model for 2D latent space when Z_DIMS = 2
def plot_latent_space(model, scale=5.0, n=25, digit_size=128, figsize=15):
    # Initialize figure with 3 channels for RGB
    figure = np.zeros((digit_size * n, digit_size * n, 3))  # Shape: (H, W, 3)

    # Construct a grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]  # Flip y-axis

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            mean, var = xi, yi
            img = generate_images_from_modelVAE(mean, var)  # Shape: (128, 128, 3)
            figure[i * digit_size : (i + 1) * digit_size, j * digit_size : (j + 1) * digit_size, :] = img  # Insert into RGB canvas

    plt.figure(figsize=(figsize, figsize))
    plt.title('VAE Latent Space Visualization')
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z1, z [0]")
    plt.ylabel("z2, z [1]")
    plt.imshow(figure)  # No cmap needed for RGB
    plt.show()


plot_latent_space(model, scale=1)
