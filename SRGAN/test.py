import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import torchsummary
import torchvision.transforms as transforms

from PIL import Image
import tqdm
import albumentations as A
import numpy as np
import zipfile
from albumentations.pytorch import ToTensorV2
import os
import wandb
import matplotlib.pyplot as plt


#----------------------------- Generator -------------------------------------#
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = self.conv(x)
        residual = self.bn(residual)
        return x + self.prelu(residual)


class Generator(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()

        self.residual_blocks = nn.Sequential(*[ResidualBlock() for _ in range(num_residual_blocks)])

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

        self.upscale = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # Tăng kích thước ảnh
            nn.PReLU(),

            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # 64x64 -> 128x128
            nn.PReLU(),
        )

        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out = self.residual_blocks(out1)
        out = self.bn(self.conv2(out)) + out1
        out = self.upscale(out)
        out = self.conv3(out)
        return self.tanh(out)


#-------------------------------- Discriminator -----------------------------------#
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 512 -> 256
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(2048 * 8 * 8, 1),  # Fully connected layer
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)  # Tiếp tục qua flatten và linear
        return out


class PortraitDataset(Dataset):
    def __init__(self, high_res_dir, low_res_dir, transform_hr=None, transform_lr=None):
        self.high_res_dir = high_res_dir
        self.low_res_dir = low_res_dir
        self.high_res_images = sorted(os.listdir(high_res_dir))
        self.low_res_images = sorted(os.listdir(low_res_dir))
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr

    def __len__(self):
        return len(self.high_res_images)

    def __getitem__(self, index):
        # Load ảnh HR và LR
        hr_path = os.path.join(self.high_res_dir, self.high_res_images[index])
        lr_path = os.path.join(self.low_res_dir, self.low_res_images[index])

        hr_image = Image.open(hr_path).convert("RGB")
        lr_image = Image.open(lr_path).convert("RGB")

        # Áp dụng các phép biến đổi (nếu có)
        if self.transform_hr:
            hr_image = self.transform_hr(hr_image)
        if self.transform_lr:
            lr_image = self.transform_lr(lr_image)

        return lr_image, hr_image


# Transform cho ảnh High-Resolution (HR)
transform_hr = transforms.Compose([
    transforms.Resize((512, 512)),  # Thay đổi kích thước nếu cần
    transforms.ToTensor(),  # Chuyển sang tensor
])

# Transform cho ảnh Low-Resolution (LR)
transform_lr = transforms.Compose([
    transforms.Resize((128, 128)),  # Thay đổi kích thước nếu cần
    transforms.ToTensor(),
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G = Generator().to(device)

G.load_state_dict(torch.load("./Checkpoints/checkpoint_epoch_20.pth")['G_state_dict'])

G.eval()


def load_image(image_path):
    # Mở ảnh và chuyển sang RGB
    image = Image.open(image_path).convert("RGB")
    image = transform_lr(image).unsqueeze(0)
    return image.to(device)


def generate_image(gener, inputimage):
    with torch.no_grad():
        fake_image = gener(inputimage)
        fake_image = fake_image.squeeze(0).cpu()
    return fake_image


def show_images(real_img, fake_img, title="Generated Image"):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.transpose(real_img.cpu().numpy(), (1, 2, 0)))
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    ax[1].imshow(np.transpose(fake_img.numpy(), (1, 2, 0)))
    ax[1].set_title(title)
    ax[1].axis("off")

    plt.show()


input_image = load_image("image1.jpg")
high_image = generate_image(G, input_image)
show_images(input_image.squeeze(0), high_image, title="Denoise")
