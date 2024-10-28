# USAGE
# python predict.py
# import the necessary packages
from pyimagesearch import config
import matplotlib.pyplot as plt
import torchvision
import torch

# load the pre-trained PGAN model
model = torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub",
                       "PGAN", model_name="celebAHQ-512", pretrained=True,
                       useGPU=config.USE_GPU)
# sample random noise vectors
(noise, _) = model.buildNoiseData(config.NUM_IMAGES)
# pass the sampled noise vectors through the pre-trained generator
with torch.no_grad():
    generatedImages = model.test(noise)
# visualize the generated images
grid = torchvision.utils.make_grid(
    generatedImages.clamp(min=-1, max=1), nrow=config.NUM_IMAGES,
    scale_each=True, normalize=True)
plt.figure(figsize=(20, 20))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
# save generated image visualizations
torchvision.utils.save_image(generatedImages.clamp(min=-1, max=1),
                             config.SAVE_IMG_PATH, nrow=config.NUM_IMAGES, scale_each=True,
                             normalize=True)
