import torch
import torch.nn as nn
import os 
from os.path import join as jn 
from PIL import Image 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms 
from tqdm import tqdm


Z_DIM = 8
# PATH_MODEL = "./output.pth"
PATH_MODEL = "./output_VAE_VGG_Feb6_10z.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"