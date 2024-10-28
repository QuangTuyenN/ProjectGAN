# import the necessary packages
import torch
import os
# define gpu or cpu usage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU = True if DEVICE == "cuda" else False
# define the number of images to generate and interpolate
NUM_IMAGES = 8
NUM_INTERPOLATION = 8
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output model output and latent
# space interpolation
SAVE_IMG_PATH = os.path.join(BASE_OUTPUT, "image_samples.png")
INTERPOLATE_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "interpolate.png"])









