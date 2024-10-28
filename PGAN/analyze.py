# USAGE
# python analyze.py
# import the necessary packages
from pyimagesearch import config
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch


def interpolate(n):
    # sample the two noise vectors z1 and z2
    (noise, _) = model.buildNoiseData(2)
    # define the step size and sample numbers in the range (0, 1) at
    # step intervals
    step = 1 / n
    lam = list(np.arange(0, 1, step))

    # initialize a tensor for storing interpolated images
    interpolatedImages = torch.zeros([n, 3, 512, 512])
    # iterate over each value of lam
    for i in range(n):
        # compute interpolated z
        zInt = (1 - lam[i]) * noise[0] + lam[i] * noise[1]

        # generate the corresponding in the images space
        with torch.no_grad():
            outputImage = model.test(zInt.reshape(-1, 512))
            interpolatedImages[i] = outputImage
    # return the interpolated images
    return interpolatedImages


# load the pre-trained PGAN model
model = torch.hub.load("facebookresearch/pytorch_GAN_zoo:hub",
                       "PGAN", model_name="celebAHQ-512", pretrained=True, useGPU=True)
# call the interpolate function
interpolatedImages = interpolate(config.NUM_INTERPOLATION)
# visualize output images
grid = torchvision.utils.make_grid(
    interpolatedImages.clamp(min=-1, max=1), scale_each=True,
    normalize=True)
plt.figure(figsize=(20, 20))
plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
# save visualizations
torchvision.utils.save_image(interpolatedImages.clamp(min=-1, max=1),
                             config.INTERPOLATE_PLOT_PATH, nrow=config.NUM_IMAGES,
                             scale_each=True, normalize=True)