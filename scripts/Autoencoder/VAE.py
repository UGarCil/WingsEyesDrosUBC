# Jan 30th, 2025
# Auth. Uriel Garcilazo Cruz
# Generate a latent space via variational autoencoder (VAE) that takes the weights and biases from the VGG model 19.

from VGG19 import Vgg19 
from constants import * 

# TRANSFER THE VAE USING THE WEIGHTS AND BIASES FROM THE VGG19 MODEL

# Load the weights from the VGG19 model
vgg19 = Vgg19()
vgg19.load_state_dict(torch.load("./vgg19-dcbb9e9d.pth"))
for param in vgg19.features.parameters():
    param.require_grad = False
    
# DD. DOWNSAMPLE_BLOCK_VGG19
# downsample = DownsampleVGG19()
# interp. a DOWNSAMPLE block represented by a pass across a subset of VGG19 feature extractor
class DownSampleVGG(nn.Module):
    def __init__(self,range_from_VGG, in_channels,out_channels, pool_kernel_size=2, pool_stride=2):
        super().__init__()
        low_bound,upper_bound = range_from_VGG
        self.conv = nn.Sequential(*list(vgg19.features.children())[low_bound:upper_bound])
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size,stride=pool_stride)
    
    def forward(self,x):
        down = self.conv(x)
        p = self.pool(down)
        return p
    
    
class VAE(nn.Module):
    def __init__(self, input_dim=3, device=device):
        super(VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            DownSampleVGG((0,4), 3, 64),
            DownSampleVGG((5,9), 64, 128)
        )

        # Calculate the flattened size after convolutions
        # After two DownSampleVGG blocks, spatial dimensions are reduced by factor of 4
        # If input is 128x128, output will be 32x32 with 128 channels
        self.flatten_size = 128 * 32 * 32  # channels * height * width

        self.mean_layer = nn.Linear(self.flatten_size, Z_DIM)
        self.logvar_layer = nn.Linear(self.flatten_size, Z_DIM)

        # Decoder layers
        self.decoder_linear = nn.Linear(Z_DIM, self.flatten_size)
        
        # Transpose convolution layers to upsample back to original size
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 32, 32)),  # Reshape to match encoder output
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )
        
    
        # decoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(2, hidden_dim2),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(hidden_dim2, hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(hidden_dim, input_dim),
        #     nn.Sigmoid()
        #     )


    def encode(self, x):
        x = self.encoder(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1)  # Flatten the tensor
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)  # More explicit flattening
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def decode(self, z):
        x = self.decoder_linear(z)
        x_hat = self.decoder(x)
        return x_hat
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        if self.decoder is not None:
            x_hat = self.decode(z)
            return x_hat, mean, log_var
        # else:
        #     # Since decoder is not implemented, just return latent variables
        #     return z, mean, log_var