import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
import random
from load_data import get_dataloader  # bạn đã có
from torchvision import models
from torchvision.models import VGG16_Weights

# ============================================================
# 1️⃣ Generator & Discriminator
# ============================================================

class Generator(nn.Module):
    def __init__(self, img_channels=1, feature_g=64):
        super(Generator, self).__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.encoder = nn.Sequential(
            block(1, feature_g),
            block(feature_g, feature_g * 2),
            block(feature_g * 2, feature_g * 4),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_channels=1, feature_d=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d * 4, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# ------------------ ReplayBuffer ------------------
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, images):
        returned = []
        for img in images:
            img = img.unsqueeze(0).detach().cpu()
            if len(self.data) < self.max_size:
                self.data.append(img)
                returned.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = img
                    returned.append(tmp)
                else:
                    returned.append(img)
        return torch.cat(returned, 0).to(images.device)

# ------------------ Perceptual Loss (VGG16) ------------------
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval().to(device)
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()
        # vgg expects [0,1] normalized by these mean/std
        self.register_buffer('vgg_mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('vgg_std', torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, x, y):
        # x,y expected in range [-1,1] (because your dataloader normalizes to mean=0.5,std=0.5)
        # convert to [0,1]
        x_01 = (x + 1.0) / 2.0
        y_01 = (y + 1.0) / 2.0
        # to 3 channels if needed
        if x_01.shape[1] == 1:
            x_01 = x_01.repeat(1,3,1,1)
            y_01 = y_01.repeat(1,3,1,1)
        # normalize to vgg
        x_v = (x_01 - self.vgg_mean) / self.vgg_std
        y_v = (y_01 - self.vgg_mean) / self.vgg_std
        fx = self.vgg(x_v)
        fy = self.vgg(y_v)
        return self.criterion(fx, fy)

# ------------------ Main training function (CycleGAN improved) ------------------
def train_cyclegan(
    csv_path,
    image_dir,
    epochs=50,
    batch_size=8,
    lr=0.0002,
    lambda_cycle=5.0,     # giảm để G không chỉ copy input
    lambda_id=0.5,
    lambda_perc=0.0,      # bật perceptual bằng đặt >0 (vd 5-20)
    replay_buffer_size=50,
    device=None
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, df = get_dataloader(csv_path, image_dir, batch_size=batch_size)

    # Models (bạn dùng Generator/Discriminator đã định nghĩa trước)
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    # Losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Perceptual (optional)
    perceptual = PerceptualLoss(device) if lambda_perc > 0 else None

    # Optimizers: giữ D lr nhỏ hơn G (giúp G không bị đàn áp)
    opt_G = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=lr, betas=(0.5, 0.999))
    opt_D_A = optim.Adam(D_A.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
    opt_D_B = optim.Adam(D_B.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))

    # Replay buffers
    fake_A_buffer = ReplayBuffer(replay_buffer_size)
    fake_B_buffer = ReplayBuffer(replay_buffer_size)

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            real_A = imgs[labels == 1]
            real_B = imgs[labels == 0]
            if len(real_A) == 0 or len(real_B) == 0:
                continue

            bsize = min(len(real_A), len(real_B))
            real_A = real_A[:bsize]
            real_B = real_B[:bsize]

            # ----------------------
            #  Train Generators
            # ----------------------
            opt_G.zero_grad()

            fake_B = G_AB(real_A)
            recov_A = G_BA(fake_B)
            fake_A = G_BA(real_B)
            recov_B = G_AB(fake_A)

            # label smoothing: real label ~0.9
            valid_B = torch.ones_like(D_B(fake_B), device=device) * 0.9
            valid_A = torch.ones_like(D_A(fake_A), device=device) * 0.9

            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid_A)

            loss_cycle_A = criterion_cycle(recov_A, real_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_G = loss_GAN_AB + loss_GAN_BA \
                     + lambda_cycle * (loss_cycle_A + loss_cycle_B) \
                     + lambda_id * (loss_id_A + loss_id_B)

            # perceptual loss (optional)
            if perceptual is not None:
                # using unpaired batches: compare fake_B with real_B (domain-level guidance)
                perc_loss_AB = perceptual(fake_B, real_B)
                perc_loss_BA = perceptual(fake_A, real_A)
                loss_G = loss_G + lambda_perc * (perc_loss_AB + perc_loss_BA)

            loss_G.backward()
            opt_G.step()

            # ----------------------
            #  Train Discriminator A
            # ----------------------
            opt_D_A.zero_grad()
            pred_real_A = D_A(real_A)
            valid_real_A = torch.ones_like(pred_real_A, device=device) * 0.9
            fake_A_buf = fake_A_buffer.push_and_pop(fake_A.detach())
            pred_fake_A = D_A(fake_A_buf)
            fake_label_A = torch.zeros_like(pred_fake_A, device=device)

            loss_real_A = criterion_GAN(pred_real_A, valid_real_A)
            loss_fake_A = criterion_GAN(pred_fake_A, fake_label_A)
            loss_D_A = 0.5 * (loss_real_A + loss_fake_A)
            loss_D_A.backward()
            opt_D_A.step()

            # ----------------------
            #  Train Discriminator B
            # ----------------------
            opt_D_B.zero_grad()
            pred_real_B = D_B(real_B)
            valid_real_B = torch.ones_like(pred_real_B, device=device) * 0.9
            fake_B_buf = fake_B_buffer.push_and_pop(fake_B.detach())
            pred_fake_B = D_B(fake_B_buf)
            fake_label_B = torch.zeros_like(pred_fake_B, device=device)

            loss_real_B = criterion_GAN(pred_real_B, valid_real_B)
            loss_fake_B = criterion_GAN(pred_fake_B, fake_label_B)
            loss_D_B = 0.5 * (loss_real_B + loss_fake_B)
            loss_D_B.backward()
            opt_D_B.step()

        # End epoch logging & save
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Loss_G: {loss_G.item():.4f} | Loss_DA: {loss_D_A.item():.4f} | Loss_DB: {loss_D_B.item():.4f}"
        )

        # Save images: convert from [-1,1] -> [0,1] for visualization
        with torch.no_grad():
            out_fake_B = (fake_B + 1.0) / 2.0
            out_real_A = (real_A + 1.0) / 2.0
            
            if fake_B is not None and fake_B.size(0) > 0:
                save_image(out_fake_B[:4], f"results/epoch_{epoch+1}_A2B.png", normalize=False)
            if out_real_A is not None and out_real_A.size(0) > 0:
                save_image(out_real_A[:4], f"results/epoch_{epoch+1}_A2B_input.png", normalize=False)

        # Checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(G_AB.state_dict(), f"checkpoints/G_AB_epoch_{epoch+1}.pth")
            torch.save(G_BA.state_dict(), f"checkpoints/G_BA_epoch_{epoch+1}.pth")
            torch.save(D_A.state_dict(), f"checkpoints/D_A_epoch_{epoch+1}.pth")
            torch.save(D_B.state_dict(), f"checkpoints/D_B_epoch_{epoch+1}.pth")

        print(f"Saved sample images and checkpoints for epoch {epoch+1}")

    print("Training finished!")


if __name__ == "__main__":
    csv_path = "Data_Entry_2017.csv"
    image_dir = "images_1"
    train_cyclegan(csv_path, image_dir, epochs=50, batch_size=8)