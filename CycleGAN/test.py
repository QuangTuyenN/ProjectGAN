# import torch
# from PIL import Image
# import numpy as np
# import torchvision.transforms as transforms

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


# config = {
#     "GAN TYPE": "CycleGAN",
#     "DISCRIMINATOR ARCHITECTURE": "PatchGAN",
#     "GENERATOR ARCHITECTURE": "UNET",
#     "DATASET": "HORSE2ZEBRA",
#     "FEATURES": 64,
#     "EPOCHS": 50,
#     "BATCH_SIZE": 1,
#     "LEARNING_RATE": 1e-5,
#     "LAMBDA_CONSISTENCY": 10,
#     "LAMBDA_IDENTITY": 5,
#     "NUM_WORKERS": 4,
#     "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# }

FEATURES = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first_block=False, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs, padding_mode='reflect', bias=True),
            nn.InstanceNorm2d(out_channels) if not first_block else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            DBlock(3, FEATURES, first_block=True, kernel_size=4, stride=2, padding=1),  # (3, 256, 256) -> (64, 128, 128)
            DBlock(FEATURES * 1, FEATURES * 2, kernel_size=4, stride=2, padding=1),  # (128, 64, 64)
            DBlock(FEATURES * 2, FEATURES * 4, kernel_size=4, stride=2, padding=1),  # (256, 32, 32)
            DBlock(FEATURES * 4, FEATURES * 8, kernel_size=4, stride=1, padding=1),  # (512, 31, 31)
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(FEATURES * 8, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),  # (1, 30, 30)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 3, 256, 256)
        x = self.model(x)
        x = self.last_layer(x)
        return x


# def test():
#     model = Discriminator()
#     x = torch.randn((5, 3, 256, 256))
#     preds = model(x)
#     print(preds.shape)
#     print(torchsummary.summary(model.to(DEVICE), (3, 256, 256)))


# Generator
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs, padding_mode="reflect") if down else
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            GBlock(channels, channels, use_act=True, kernel_size=3, stride=1, padding=1),
            GBlock(channels, channels, use_act=False, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, FEATURES, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),  # (3, 256, 256) -> (64, 256, 256)
            nn.ReLU(inplace=True),
        )

        self.down_block = nn.Sequential(
            GBlock(FEATURES * 1, FEATURES * 2, kernel_size=3, stride=2, padding=1),  # (128, 128, 128)
            GBlock(FEATURES * 2, FEATURES * 4, kernel_size=3, stride=2, padding=1),  # (256, 64, 64)
        )

        self.residual_block = nn.Sequential(
            *[ResidualBlock(FEATURES * 4) for _ in range(num_residuals)]
        )

        self.up_block = nn.Sequential(
            GBlock(FEATURES * 4, FEATURES * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (128, 128, 128)
            GBlock(FEATURES * 2, FEATURES * 1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (64, 256, 256)
        )

        self.last = nn.Sequential(
            nn.Conv2d(FEATURES, 3, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),  # (3, 256, 256)
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down_block(x)
        x = self.residual_block(x)
        x = self.up_block(x)
        x = self.last(x)
        return x


# Load lại Generator
gen_Z = Generator().to(DEVICE)  # Chuyển từ horse -> zebra
gen_H = Generator().to(DEVICE)  # Chuyển từ zebra -> horse

# Load checkpoint
checkpoint = torch.load("./Checkpoints/checkpoint_epoch_48.pt", map_location=DEVICE)
gen_H.load_state_dict(checkpoint['gen_H_state_dict'])
gen_Z.load_state_dict(checkpoint['gen_Z_state_dict'])

# Chuyển mô hình sang chế độ đánh giá
gen_H.eval()
gen_Z.eval()


def load_image(image_path):
    # Mở ảnh và chuyển sang RGB
    image = Image.open(image_path).convert("RGB")

    # Áp dụng các phép biến đổi giống như lúc training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    image = transform(image).unsqueeze(0)  # Thêm batch dimension
    return image.to(DEVICE)


def generate_image(generator, input_image):
    with torch.no_grad():  # Tắt gradient để tăng tốc
        fake_image = generator(input_image)
        fake_image = fake_image.squeeze(0).cpu()  # Bỏ batch dimension và chuyển về CPU
        fake_image = (fake_image * 0.5 + 0.5).clamp(0, 1)  # Undo normalize [-1,1] -> [0,1]
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


# Load ảnh ngựa để chuyển thành zebra
input_image = load_image("zebra1.jpg")

# Dùng Generator để tạo ảnh zebra
fake_zebra = generate_image(gen_H, input_image)

# Hiển thị ảnh đầu vào và ảnh đã chuyển đổi
show_images(input_image.squeeze(0), fake_zebra, title="Horse to Zebra")




