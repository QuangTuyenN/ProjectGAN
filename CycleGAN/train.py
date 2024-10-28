import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import torch.nn.functional as F
import torchsummary

from PIL import Image
import tqdm
import albumentations as A
import numpy as np
import zipfile
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import os
import wandb

config = {
    "GAN TYPE": "CycleGAN",
    "DISCRIMINATOR ARCHITECTURE": "PatchGAN",
    "GENERATOR ARCHITECTURE": "UNET",
    "DATASET": "HORSE2ZEBRA",
    "FEATURES": 64,
    "EPOCHS": 50,
    "BATCH_SIZE": 1,
    "LEARNING_RATE": 1e-5,
    "LAMBDA_CONSISTENCY": 10,
    "LAMBDA_IDENTITY": 5,
    "NUM_WORKERS": 4,
    "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}
wandb.init(project="Horse_vs_Zebra_Transfer", config=config)
config = wandb.config

transform = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2()
    ],
    additional_targets={"image0": "image"},
)


class HorseZebraDataset(Dataset):
    def __init__(self, horse_path, zebra_path, transform=None):
        self.horse_path = horse_path
        self.zebra_path = zebra_path
        self.transform = transform

        self.horse_images = os.listdir(horse_path)
        self.zebra_images = os.listdir(zebra_path)

        self.horse_len = len(self.horse_images)
        self.zebra_len = len(self.zebra_images)
        self.length_dataset = max(self.horse_len, self.zebra_len)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        horse_image = self.horse_images[index % self.horse_len]
        zebra_image = self.zebra_images[index % self.zebra_len]

        horse_image_path = os.path.join(self.horse_path, horse_image)
        zebra_image_path = os.path.join(self.zebra_path, zebra_image)

        horse_array = np.array(Image.open(horse_image_path).convert("RGB"))
        zebra_array = np.array(Image.open(zebra_image_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=horse_array, image0=zebra_array)
            horse_array = augmentations["image"]
            zebra_array = augmentations["image0"]

        return horse_array, zebra_array


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
            DBlock(3, config.FEATURES, first_block=True, kernel_size=4, stride=2, padding=1),  # (3, 256, 256) -> (64, 128, 128)
            DBlock(config.FEATURES * 1, config.FEATURES * 2, kernel_size=4, stride=2, padding=1),  # (128, 64, 64)
            DBlock(config.FEATURES * 2, config.FEATURES * 4, kernel_size=4, stride=2, padding=1),  # (256, 32, 32)
            DBlock(config.FEATURES * 4, config.FEATURES * 8, kernel_size=4, stride=1, padding=1),  # (512, 31, 31)
        )
        self.last_layer = nn.Sequential(
            nn.Conv2d(config.FEATURES * 8, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),  # (1, 30, 30)
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
#     print(torchsummary.summary(model.to(config.DEVICE), (3, 256, 256)))


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
            nn.Conv2d(3, config.FEATURES, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),  # (3, 256, 256) -> (64, 256, 256)
            nn.ReLU(inplace=True),
        )

        self.down_block = nn.Sequential(
            GBlock(config.FEATURES * 1, config.FEATURES * 2, kernel_size=3, stride=2, padding=1),  # (128, 128, 128)
            GBlock(config.FEATURES * 2, config.FEATURES * 4, kernel_size=3, stride=2, padding=1),  # (256, 64, 64)
        )

        self.residual_block = nn.Sequential(
            *[ResidualBlock(config.FEATURES * 4) for _ in range(num_residuals)]
        )

        self.up_block = nn.Sequential(
            GBlock(config.FEATURES * 4, config.FEATURES * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (128, 128, 128)
            GBlock(config.FEATURES * 2, config.FEATURES * 1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (64, 256, 256)
        )

        self.last = nn.Sequential(
            nn.Conv2d(config.FEATURES, 3, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),  # (3, 256, 256)
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down_block(x)
        x = self.residual_block(x)
        x = self.up_block(x)
        x = self.last(x)
        return x


# def test():
#     model = Generator()
#     x = torch.randn((5, 3, 256, 256))
#     preds = model(x)
#     print(preds.shape)
#     print(torchsummary.summary(model.to(config.DEVICE), (3, 256, 256)))

def train():
    # Losses
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    # Instantiate dataset and dataloader
    HORSE_PATH = "./Data/trainA"
    ZEBRA_PATH = "./Data/trainB"
    CHECKPOINT_DIR = "./Checkpoints"
    dataset = HorseZebraDataset(horse_path=HORSE_PATH, zebra_path=ZEBRA_PATH, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    wandb.init(project="Horse_vs_Zebra_Transfer", config=config)

    # Instantiate models
    disc_H = Discriminator().to(config.DEVICE)  # Classify horse images as real or fake
    disc_Z = Discriminator().to(config.DEVICE)  # Classify zebra images as real or fake
    gen_H = Generator().to(config.DEVICE)  # Generate horse images
    gen_Z = Generator().to(config.DEVICE)  # Generate zebra images

    # Optimizers
    optim_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    optim_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    last_epoch = -1

    load_checkpoint = False
    if load_checkpoint:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{last_epoch}.pt')
        checkpoint = torch.load(checkpoint_path)

        # Load the models
        gen_H.load_state_dict(checkpoint['gen_H_state_dict'])
        gen_Z.load_state_dict(checkpoint['gen_Z_state_dict'])
        disc_H.load_state_dict(checkpoint['disc_H_state_dict'])
        disc_Z.load_state_dict(checkpoint['disc_Z_state_dict'])

        # Load the optimizers
        optim_gen.load_state_dict(checkpoint['optim_gen_state_dict'])
        optim_disc.load_state_dict(checkpoint['optim_disc_state_dict'])

        # Get other information from the checkpoint if needed
        last_epoch = checkpoint['last_epoch']

    # Main train

    d_scaler = torch.cuda.amp.GradScaler()
    g_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(last_epoch + 1, config.EPOCHS):
        loop = tqdm.tqdm(loader, leave=True)
        loop.set_description(f"Epoch {epoch}")
        for step, (horse, zebra) in enumerate(loop):
            horse = horse.to(config.DEVICE)
            zebra = zebra.to(config.DEVICE)

            ### Discriminators ###
            with torch.cuda.amp.autocast():
                # Horse
                fake_horse = gen_H(zebra)
                real_H_preds = disc_H(horse)
                fake_H_preds = disc_H(fake_horse.detach())

                D_real_H_loss = mse(real_H_preds, torch.ones_like(real_H_preds))
                D_fake_H_loss = mse(fake_H_preds, torch.zeros_like(fake_H_preds))
                D_H_loss = D_real_H_loss + D_fake_H_loss

                # Zebra
                fake_zebra = gen_Z(horse)
                real_Z_preds = disc_Z(zebra)
                fake_Z_preds = disc_Z(fake_zebra.detach())

                D_real_Z_loss = mse(real_Z_preds, torch.ones_like(real_Z_preds))
                D_fake_Z_loss = mse(fake_Z_preds, torch.zeros_like(fake_Z_preds))
                D_Z_loss = D_real_Z_loss + D_fake_Z_loss

                # Put those 2 losses together
                D_loss = (D_H_loss + D_Z_loss) / 2

            # Backpropagation on disciminators
            optim_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(optim_disc)
            d_scaler.update()

            ### Train generators ###
            with torch.cuda.amp.autocast():
                fake_H_preds = disc_H(fake_horse)
                fake_Z_preds = disc_Z(fake_zebra)
                G_H_loss = mse(fake_H_preds, torch.ones_like(fake_H_preds))
                G_Z_loss = mse(fake_Z_preds, torch.ones_like(fake_Z_preds))

                # Cycle Loss
                cycle_horse = gen_H(fake_zebra)
                cycle_zebra = gen_Z(fake_horse)
                consistency_H_loss = l1(horse, cycle_horse)
                consistency_Z_loss = l1(zebra, cycle_zebra)

                # Identity loss
                # identity_horse = gen_H(horse)
                # identity_zebra = gen_Z(zebra)
                # identity_H_loss = l1(horse, identity_horse)
                # identity_Z_loss = l1(zebra, identity_zebra)

                # Put them together
                G_loss = (
                        G_H_loss +
                        G_Z_loss +
                        consistency_H_loss * config.LAMBDA_CONSISTENCY +
                        consistency_Z_loss * config.LAMBDA_CONSISTENCY
                    # identity_H_loss * config.LAMBDA_IDENTITY +
                    # identity_Z_loss * config.LAMBDA_IDENTITY
                )

            # Backpropagation on generators
            optim_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(optim_gen)
            g_scaler.update()

            if step % 50 == 0:
                with torch.no_grad():
                    wandb.log({
                        "Discriminator Horse": D_H_loss.item(),
                        "Discriminator Zebra": D_Z_loss.item(),
                        "Generator Horse": G_H_loss.item(),
                        "Generator Zebra": G_Z_loss.item(),
                        "Horse Consistency": consistency_H_loss.item(),
                        "Zebra Consistency": consistency_Z_loss.item(),
                        "Batch index": step
                    })

                    row1_grid = vutils.make_grid([horse[0], fake_zebra[0], cycle_horse[0]], normalize=True).cpu().permute(1, 2, 0).numpy()
                    row2_grid = vutils.make_grid([zebra[0], fake_horse[0], cycle_zebra[0]], normalize=True).cpu().permute(1, 2, 0).numpy()
                    merged_grid = np.concatenate((row1_grid, row2_grid), axis=0)

                    wandb.log({
                        "Generated images": wandb.Image(merged_grid),
                    })

        last_epoch = epoch
        # Save checkpoint after each epoch
        # current_run_name = wandb.run.name
        # checkpoint_path = os.path.join(CHECKPOINT_DIR, current_run_name)
        # os.mkdir(checkpoint_path)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'last_epoch': epoch,
            'gen_H_state_dict': gen_H.state_dict(),
            'gen_Z_state_dict': gen_Z.state_dict(),
            'disc_H_state_dict': disc_H.state_dict(),
            'disc_Z_state_dict': disc_Z.state_dict(),
            'optim_gen_state_dict': optim_gen.state_dict(),
            'optim_disc_state_dict': optim_disc.state_dict(),
        }, checkpoint_path)
        print("==> Saved checkpoint.")

if __name__ == '__main__':
    train()