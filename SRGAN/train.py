import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import tqdm


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


def save_checkpoint(G, D, optimizer_G, optimizer_D, epoch, folder="Checkpoints"):
    # Tạo tên file với epoch để tránh ghi đè
    checkpoint_path = os.path.join(folder, f"checkpoint_epoch_{epoch}.pth")
    # Lưu trạng thái vào file
    checkpoint = {
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}.")


def train():
    G = Generator()
    D = Discriminator()

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(D.parameters(), lr=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.to(device)
    D.to(device)

    # Đường dẫn đến thư mục chứa ảnh
    high_res_dir = "ImageData"
    low_res_dir = "LowImageData"

    # Khởi tạo Dataset
    dataset = PortraitDataset(
        high_res_dir=high_res_dir,
        low_res_dir=low_res_dir,
        transform_hr=transform_hr,
        transform_lr=transform_lr
    )

    # Khởi tạo DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # # Kiểm tra một batch dữ liệu
    # for lr_images, hr_images in dataloader:
    #     print(f"Low-Resolution Batch Shape: {lr_images.shape}")
    #     print(f"High-Resolution Batch Shape: {hr_images.shape}")
    #     break

    num_epochs = 40
    for epoch in range(1, num_epochs + 1):
        loop = tqdm.tqdm(dataloader, leave=True)
        loop.set_description(f"Epoch {epoch}")
        for low_res_imgs, high_res_imgs in loop:
            low_res_imgs, high_res_imgs = low_res_imgs.to(device), high_res_imgs.to(device)

            # Huấn luyện Discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones(low_res_imgs.size(0), 1).to(device)
            fake_labels = torch.zeros(low_res_imgs.size(0), 1).to(device)

            real_loss = criterion(D(high_res_imgs), real_labels)

            # print("low res imgs shape: ", low_res_imgs.shape)
            fake_imgs = G(low_res_imgs)
            # print("fake imgs shape: ", fake_imgs.shape)
            fake_loss = criterion(D(fake_imgs.detach()), fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Huấn luyện Generator
            optimizer_G.zero_grad()
            g_loss = criterion(D(fake_imgs), real_labels)
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch + 1}/100], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")
        # Lưu checkpoint mỗi 10 epoch
        if epoch % 10 == 0:
            save_checkpoint(G, D, optimizer_G, optimizer_D, epoch)
    # Lưu checkpoint cuối cùng
    save_checkpoint(G, D, optimizer_G, optimizer_D, num_epochs)


if __name__ == "__main__":
    train()


