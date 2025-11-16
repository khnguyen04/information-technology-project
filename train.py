import os
from PIL import Image
import math
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import lpips

# ---------------------------------------------------
# Utility
# ---------------------------------------------------
def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------
# Dataset 
# ---------------------------------------------------
class XrayBBDMDataset(Dataset):
    def __init__(self, root="data", img_size=256):
        self.root = root
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(".png")
        ]

        if len(self.files) == 0:
            raise ValueError("No PNG images found in the data/ folder")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # 1-channel grayscale normalized
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")  # X-ray -> grayscale
        return self.transform(img)

# ---------------------------------------------------
# Residual Block
# ---------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)

# ---------------------------------------------------
# Time embedding for diffusion
# ---------------------------------------------------
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :] * 2 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] != self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.lin(emb)

# ---------------------------------------------------
# TinyUNet for Type 3 Self-Consistency BBDM
# ---------------------------------------------------
class TinyUNetType3(nn.Module):
    def __init__(self, in_ch=1, base=64):
        super().__init__()
        # Encoder
        self.enc1 = ResidualBlock(in_ch, base)        # 64
        self.enc2 = ResidualBlock(base, base * 2)     # 128
        self.enc3 = ResidualBlock(base * 2, base * 4) # 256

        # Middle
        self.mid = ResidualBlock(base * 4, base * 4)

        # Decoder
        self.dec3 = ResidualBlock(base * 4 + base * 2, base * 2)
        self.dec2 = ResidualBlock(base * 2 + base, base)
        self.dec1 = ResidualBlock(base + base, base)

        # Upsample + Output
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.out = nn.Conv2d(base, 1, 1)  # 1 channel output

        # Time embedding
        self.time_mlp = TimeEmbedding(base * 4)

    def forward(self, x0, t):
        """
        x0: (B,1,H,W) input grayscale
        t: (B,1) normalized time 0~1
        """
        # Encoder
        e1 = self.enc1(x0)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))

        # Time embedding
        te = self.time_mlp(t).view(t.size(0), -1, 1, 1)
        e3 = e3 + te

        # Middle
        m = self.mid(e3)

        # Decoder
        d3 = torch.cat([self.up(m), e2], dim=1)
        d3 = self.dec3(d3)

        d2 = torch.cat([self.up(d3), e1], dim=1)
        d2 = self.dec2(d2)

        d1 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)

# ---------------------------------------------------
# Brownian Bridge: x0 -> xt
# ---------------------------------------------------
def brownian_bridge_sample(x0, t):
    """
    Generate Brownian Bridge noisy image xt from x0.
    
    Args:
        x0: (B,1,H,W) input grayscale image
        t: (B,1) time in [0,1]
        
    Returns:
        xt: noisy image
        noise: the random noise added
    """
    t_ = t.view(-1, 1, 1, 1)
    noise = torch.randn_like(x0)
    sigma = torch.sqrt(t_ * (1 - t_) + 1e-12)  # noise peaks at t=0.5
    xt = x0 + sigma * noise
    return xt, noise


# ---------------------------------------------------
# Estimate x0 from xt and predicted noise
# ---------------------------------------------------
def predict_x0_from_xt(xt, pred_noise, t):
    """
    Reconstruct x0 from xt and predicted noise.
    
    Args:
        xt: noisy image
        pred_noise: model's predicted noise
        t: time in [0,1]
        
    Returns:
        x0_pred: predicted clean image
    """
    t_ = t.view(-1, 1, 1, 1)
    sigma = torch.sqrt(t_ * (1 - t_) + 1e-12)
    x0_pred = xt - sigma * pred_noise
    return x0_pred

# ---------------------------------------------------
# Training function
# ---------------------------------------------------
def train_bbdm(
    dataset,
    model,
    epochs=50,
    batch_size=8,
    lr=1e-4,
    save_dir="checkpoints",
    device=None
):
    """
    Train Brownian Bridge Diffusion Model (Type 3) on grayscale X-ray dataset.
    
    Args:
        dataset: torch.utils.data.Dataset
        model: TinyUNetType3 instance
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        save_dir: directory to save checkpoints
        device: "cuda" or "cpu" (default auto)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x0 in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            x0 = x0.to(device)  # (B,1,H,W)
            B = x0.size(0)
            # Sample random times
            t = torch.rand(B, 1, device=device)  # t in [0,1]

            # Brownian Bridge noisy image
            xt, noise = brownian_bridge_sample(x0, t)

            # Model prediction
            pred_noise = model(xt, t)

            # Loss: MSE between predicted noise and true noise
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B

        print(f"[Epoch {epoch}/{epochs}] Average Loss: {running_loss / len(dataset):.6f}")
        # Save checkpoint every epoch
        save_path = os.path.join(save_dir, f"bbdm_epoch{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"Epoch {epoch} completed. Checkpoint saved to {save_path}")

# ---------------------------------------------------
# Test function
# ---------------------------------------------------
from torchvision.utils import save_image
def test_bbdm(
    model,
    checkpoint_path,
    x0,                # tensor (1,H,W) grayscale, giá trị [-1,1]
    num_samples=5,
    save_dir="test_results",
    device=None
):
    """
    Generate multi-hypothesis predictions from a single X-ray image using trained BBDM.
    
    Args:
        model: TinyUNetType3 instance
        checkpoint_path: path to saved model checkpoint
        x0: tensor (1,H,W) grayscale image in [-1,1]
        num_samples: number of multi-hypothesis outputs
        save_dir: folder to save results
        device: "cuda" or "cpu"
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    x0 = x0.to(device).unsqueeze(0)  # add batch dim -> (1,1,H,W)
    save_image((x0 + 1) / 2, os.path.join(save_dir, "x0.png"))

    generated = []
    with torch.no_grad():
        for i in range(num_samples):
            # Sample random time t
            t = torch.rand(1,1, device=device)
            # Brownian Bridge noisy image
            xt, _ = brownian_bridge_sample(x0, t)
            # Predict noise
            pred_noise = model(xt, t)
            # Reconstruct x0
            x0_pred = predict_x0_from_xt(xt, pred_noise, t)
            generated.append(x0_pred.clamp(-1,1))  # clamp to [-1,1]

            # Save noisy image xt
            noisy_path = os.path.join(save_dir, f"hypothesis_{i+1}_noisy.png")
            save_image((xt + 1) / 2, noisy_path)  # [-1,1] -> [0,1]

            # Save each hypothesis
            save_path = os.path.join(save_dir, f"hypothesis_{i+1}.png")
            save_image((x0_pred + 1) / 2, save_path)  # [-1,1] -> [0,1]
            print(f"Hypothesis {i+1} saved to {save_path}")

    return torch.cat(generated, dim=0)  # (num_samples,1,H,W)

if __name__ == "__main__":
    def train():
        # ---------------------------
        # Config
        # ---------------------------
        EPOCHS = 5
        BATCH_SIZE = 1
        LEARNING_RATE = 1e-4
        SAVE_DIR = "checkpoints"
        IMG_SIZE = 256
        BASE_CH = 64
        DATA_DIR = "images_1"

        # ---------------------------
        # Dataset
        # ---------------------------
        dataset = XrayBBDMDataset(root=DATA_DIR, img_size=IMG_SIZE)
        print(f"Found {len(dataset)} images in {DATA_DIR}")

        # ---------------------------
        # Model
        # ---------------------------
        model = TinyUNetType3(in_ch=1, base=BASE_CH)

        # ---------------------------
        # Train
        # ---------------------------
        train_bbdm(
            dataset=dataset,
            model=model,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            save_dir=SAVE_DIR,
            device=default_device()
        )

    def test():
        # Load 1 X-ray test image
        img = Image.open("images_3/00003926_000.png").convert("L")
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])
        x0 = transform(img)  # (1,H,W)

        # Tạo model
        model = TinyUNetType3(in_ch=1, base=64)

        # Test
        samples = test_bbdm(
            model=model,
            checkpoint_path="checkpoints/bbdm_epoch5.pt",
            x0=x0,
            num_samples=5,
            save_dir="test_results"
        )

    # train()
    test()