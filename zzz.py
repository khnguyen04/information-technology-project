import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from tqdm import tqdm

# ============================================================
# CONSTANTS
# ============================================================
IMG_CHANNELS = 1
CONDITION_DIM = 14
CONTEXT_DIM = 768
TIME_EMBED_DIM = 512
UNET_BASE_CHANNELS = 64
NUM_TIMESTEPS = 1000
LEARNING_RATE = 1e-4
EPOCHS = 10
# SCALE = 0.18215
LATENT_CHANNELS = 64
IMG_CHANNELS = 1
IMG_SIZE = 256

# ============================================================
# VAE ENCODER
# ============================================================
class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = 64

        self.down1 = self._blk(IMG_CHANNELS, base)        # 256 ->128
        self.down2 = self._blk(base, base*2)              # 128 -> 64
        self.down3 = self._blk(base*2, base*4)            # 64  -> 32
        self.down4 = self._blk(base*4, base*8)            # 32  -> 16
        self.down5 = self._blk(base*8, LATENT_CHANNELS*2) # 16  -> 8

        # final: output channels = LATENT_CHANNELS*2
        # split: mu , logvar (mỗi cái LATENT_CHANNELS)

    def _blk(self, ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 4, stride=2, padding=1),
            nn.BatchNorm2d(oc),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        h = self.down1(x)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = self.down5(h)  # shape: [B, LATENT_CHANNELS*2, 8, 8]

        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar


# ============================================================
# VAE DECODER
# ============================================================
class VAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = 64

        self.up1 = self._blk(LATENT_CHANNELS, base*8)   # 8  -> 16
        self.up2 = self._blk(base*8, base*4)            # 16 -> 32
        self.up3 = self._blk(base*4, base*2)            # 32 -> 64
        self.up4 = self._blk(base*2, base)              # 64 ->128
        self.up5 = self._blk(base, base//2)             #128 ->256

        self.out = nn.Conv2d(base//2, IMG_CHANNELS, 3, padding=1)

    def _blk(self, ic, oc):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.BatchNorm2d(oc),
            nn.SiLU(),
        )

    def forward(self, z):
        h = self.up1(z)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        h = self.up5(h)

        x = self.out(h)
        return torch.tanh(x)


# ============================================================
# CONDITION MLP -> 768D
# ============================================================
class ConditionProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(CONDITION_DIM, CONTEXT_DIM*2),
            nn.GELU(),
            nn.Linear(CONTEXT_DIM*2, CONTEXT_DIM),
        )

    def forward(self, cond):
        ctx = self.proj(cond)
        return ctx.unsqueeze(1)  # [B,1,768]

# ============================================================
# CROSS ATTENTION
# ============================================================
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, channels, context_dim=CONTEXT_DIM, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(context_dim, channels, bias=False)
        self.to_v = nn.Linear(context_dim, channels, bias=False)
        self.to_out = nn.Linear(channels, channels)
        self.norm = nn.GroupNorm(32, channels)
    
    def forward(self, x, context):
        B, C, H, W = x.shape
        x_in = x
        
        # Normalize và reshape
        x = self.norm(x).permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Query từ features
        q = self.to_q(x)  # (B, N, C)
        
        # Key, Value từ context - thêm kiểm tra kích thước
        if context.dim() == 2:
            context = context.unsqueeze(1)  # (B, 1, context_dim) -> (B, M, context_dim)
        
        k = self.to_k(context)  # (B, M, C)
        v = self.to_v(context)  # (B, M, C)
        
        # Split heads
        q = q.view(B, -1, self.heads, C // self.heads).transpose(1, 2)  # (B, heads, N, dim)
        k = k.view(B, -1, self.heads, C // self.heads).transpose(1, 2)  # (B, heads, M, dim)
        v = v.view(B, -1, self.heads, C // self.heads).transpose(1, 2)  # (B, heads, M, dim)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, N, M)
        attn = attn.softmax(dim=-1)
        
        # Output
        out = torch.matmul(attn, v)  # (B, heads, N, dim)
        out = out.transpose(1, 2).contiguous().view(B, -1, C)  # (B, N, C)
        out = self.to_out(out)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return x_in + out

# ============================================================
# CONDITIONAL UNET IMAGE-TO-IMAGE
# ============================================================
# Helper blocks
class SiLU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride, padding=1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.act1 = SiLU()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act2 = SiLU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        if in_ch != out_ch:
            self.nin_shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.nin_shortcut = nn.Identity()
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        # add time embedding
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        return h + self.nin_shortcut(x)

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
    def forward(self, x): return self.op(x)

# class Upsample(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.op = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
#     def forward(self, x): return self.op(x)

class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
    
# ============================================================
# CONDITIONAL UNET IMAGE-TO-IMAGE (FIXED + STABLE)
# ============================================================
class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels=LATENT_CHANNELS*2,
        out_channels=LATENT_CHANNELS,
        base_channels=128,
        channel_mults=(1,2,3,4),
        time_emb_dim=TIME_EMBED_DIM,
        context_dim=CONTEXT_DIM,
        heads=8
    ):
        super().__init__()

        # time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim*4),
            SiLU(),
            nn.Linear(time_emb_dim*4, time_emb_dim)
        )

        # FIX: Tính toán channels chính xác
        chs = [base_channels * mult for mult in channel_mults]  # [128, 256, 384, 512]
        
        # initial conv
        self.conv_in = nn.Conv2d(in_channels, chs[0], 3, padding=1)
        self.condition_proj = ConditionProjector()
        
        # encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        in_ch = chs[0]
        for i, out_ch in enumerate(chs):
            self.enc_blocks.append(
                nn.ModuleList([
                    ResBlock(in_ch, out_ch, time_emb_dim),
                    ResBlock(out_ch, out_ch, time_emb_dim),
                    MultiHeadCrossAttention(out_ch, context_dim=context_dim, heads=heads)
                ])
            )
            
            if i < len(chs) - 1:
                next_ch = chs[i+1]
                self.downs.append(Downsample(out_ch, next_ch))
                in_ch = next_ch
            else:
                in_ch = out_ch

        # bottleneck
        bott_ch = chs[-1]
        self.mid_block1 = ResBlock(bott_ch, bott_ch, time_emb_dim)
        self.mid_attn = MultiHeadCrossAttention(bott_ch, context_dim=context_dim, heads=heads)
        self.mid_block2 = ResBlock(bott_ch, bott_ch, time_emb_dim)

        # FIX: Decoder - tính toán channels chính xác
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        # Reverse channels cho decoder (bỏ bottleneck channel)
        dec_chs = list(reversed(chs))  # [512, 384, 256, 128]
        
        in_ch = bott_ch  # 512
        for i in range(len(dec_chs) - 1):
            out_ch = dec_chs[i+1]  # 384, 256, 128
            
            # Upsample: in_ch -> out_ch
            self.ups.append(Upsample(in_ch, out_ch))
            
            # FIX: ResBlock input = out_ch (upsampled) + chs[-(i+1)] (skip) 
            skip_ch = chs[-(i+2)]  # skip connection channels tương ứng
            
            # Input channels cho ResBlock đầu tiên = out_ch + skip_ch
            self.dec_blocks.append(
                nn.ModuleList([
                    ResBlock(out_ch + skip_ch, out_ch, time_emb_dim),
                    ResBlock(out_ch, out_ch, time_emb_dim),
                    MultiHeadCrossAttention(out_ch, context_dim=context_dim, heads=heads)
                ])
            )
            in_ch = out_ch

        # final output
        self.norm_out = nn.GroupNorm(32, chs[0])
        self.act_out = SiLU()
        self.conv_out = nn.Conv2d(chs[0], out_channels, 3, padding=1)

    def forward(self, z_t, t_emb, context, z_cond):
        # Kiểm tra và chuẩn hóa context
        if context.dim() == 2:
            context = context.unsqueeze(1)
        
        # combine latent + condition
        h = torch.cat([z_t, z_cond], dim=1)
        t = self.time_mlp(t_emb)

        # encoder
        h = self.conv_in(h)
        skips = []

        for i, (r1, r2, attn) in enumerate(self.enc_blocks):
            h = r1(h, t)
            h = r2(h, t)
            h = attn(h, context)
            skips.append(h)  # lưu skip connection

            if i < len(self.downs):
                h = self.downs[i](h)

        # bottleneck
        h = self.mid_block1(h, t)
        h = self.mid_attn(h, context)
        h = self.mid_block2(h, t)

        # FIX: decoder với skip connections chính xác
        for i, (r1, r2, attn) in enumerate(self.dec_blocks):
            # Upsample
            h = self.ups[i](h)
            
            # Lấy skip connection tương ứng
            # skips: [128@8x8, 256@4x4, 384@2x2, 512@1x1]
            # decoder: bắt đầu từ 512@1x1 -> 384@2x2 -> 256@4x4 -> 128@8x8
            skip_idx = len(skips) - i - 2  # tính index skip tương ứng
            skip = skips[skip_idx]
            
            # Đảm bảo spatial dimensions khớp
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="nearest")
            
            # Concatenate với skip connection
            h = torch.cat([h, skip], dim=1)
            
            h = r1(h, t)
            h = r2(h, t)
            h = attn(h, context)

        # final output
        h = self.norm_out(h)
        h = self.act_out(h)
        return self.conv_out(h)


# ============================================================
# GAUSSIAN DIFFUSION IMAGE-TO-IMAGE
# ============================================================
class GaussianDiffusion(nn.Module):
    def __init__(self, model, vae, vae_dec, timesteps=1000):
        super().__init__()
        self.model = model
        self.vae = vae
        self.vae_dec = vae_dec
        self.timesteps = timesteps
        for p in self.vae.parameters(): p.requires_grad=False
        for p in self.vae_dec.parameters(): p.requires_grad=False
        self.vae.eval()
        self.vae_dec.eval()

        betas = torch.linspace(1e-4,0.02,timesteps)
        alphas = 1.-betas
        acp = torch.cumprod(alphas,dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_acp", torch.sqrt(acp))
        self.register_buffer("sqrt_1m_acp", torch.sqrt(1-acp))

    # timestep embedding
    def timestep_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32).to(t.device) / half)
        args = t[:,None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim%2==1:
            emb = torch.cat([emb, torch.zeros_like(emb[:,:1])], dim=-1)
        return emb

    # encode x
    def get_latent(self, x):
        mu, _ = self.vae(x)
        return mu

    # noise
    def noise(self, z0, t):
        alpha = self.sqrt_acp[t].view(-1,1,1,1)
        sigma = self.sqrt_1m_acp[t].view(-1,1,1,1)
        eps = torch.randn_like(z0)
        z_t = alpha*z0 + sigma*eps
        return z_t, eps

    # forward training
    # def forward(self, x_target, x_cond, cond_vec):
    #     B = x_target.size(0)
    #     device = x_target.device
    #     z0 = self.get_latent(x_target) 
    #     z_cond = self.get_latent(x_cond) 

    #     t = torch.randint(0,self.timesteps,(B,), device=device).long()
    #     z_t, eps_true = self.noise(z0, t)
    #     t_emb = self.timestep_embedding(t, TIME_EMBED_DIM)
    #     context = self.model.condition_proj(cond_vec)

    #     eps_pred = self.model(z_t, t_emb, context, z_cond)
    #     loss = F.mse_loss(eps_pred, eps_true)
    #     return loss
    def forward(self, x_target, x_cond, cond_vec, drop_prob=0.15):
        """
        Training with classifier-free guidance support.
        
        Args:
            x_target: Target images to generate
            x_cond: Conditioning images
            cond_vec: Conditioning vectors
            drop_prob: Probability to drop conditions (for CFG training)
        """
        B = x_target.size(0)
        device = x_target.device
        
        z0 = self.get_latent(x_target) 
        z_cond = self.get_latent(x_cond)

        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        z_t, eps_true = self.noise(z0, t)
        t_emb = self.timestep_embedding(t, TIME_EMBED_DIM)
        
        # Project condition vector
        context = self.model.condition_proj(cond_vec)
        
        # Randomly drop conditions during training (classifier-free guidance)
        # Create mask for each sample in batch
        drop_mask = (torch.rand(B, device=device) < drop_prob).float()
        
        # Zero out conditions for dropped samples
        context_masked = context * (1 - drop_mask).view(B, 1, 1)
        z_cond_masked = z_cond * (1 - drop_mask).view(B, 1, 1, 1)

        eps_pred = self.model(z_t, t_emb, context_masked, z_cond_masked)
        loss = F.mse_loss(eps_pred, eps_true)
        
        return loss

    # p_sample
    # @torch.no_grad()
    # def p_sample(self, z_t, z_cond, t_idx, context, scale):
    #     device = z_t.device
    #     B = z_t.size(0)
    #     alpha_bar = self.alphas_cumprod[t_idx].to(device)
    #     sqrt_alpha = self.sqrt_acp[t_idx].to(device)
    #     sqrt_one_minus = self.sqrt_1m_acp[t_idx].to(device)
    #     t_vec = torch.full((B,), t_idx, device=device, dtype=torch.long)
    #     t_emb = self.timestep_embedding(t_vec, TIME_EMBED_DIM)
    #     z_in = z_t
    #     eps = self.model(z_t, t_emb, context, z_cond)
    #     z0_est = (z_t - sqrt_one_minus*eps)/sqrt_alpha
    #     if t_idx==0:
    #         return z0_est
    #     alpha_prev = self.alphas_cumprod[t_idx-1].to(device)
    #     return torch.sqrt(alpha_prev)*z0_est + torch.sqrt(1-alpha_prev)*torch.randn_like(z_t)
    @torch.no_grad()
    def p_sample(self, z_t, z_cond, t_idx, context, guidance_scale):
        """
        Single denoising step with classifier-free guidance.
        
        Args:
            z_t: Noisy latent at timestep t
            z_cond: Conditioning latent
            t_idx: Current timestep index
            context: Conditioning context
            guidance_scale: Guidance strength (1.0 = no guidance, higher = stronger)
        """
        device = z_t.device
        B = z_t.size(0)
        
        sqrt_alpha = self.sqrt_acp[t_idx].to(device)
        sqrt_one_minus = self.sqrt_1m_acp[t_idx].to(device)
        
        t_vec = torch.full((B,), t_idx, device=device, dtype=torch.long)
        t_emb = self.timestep_embedding(t_vec, TIME_EMBED_DIM)
        
        # Conditional prediction
        eps_cond = self.model(z_t, t_emb, context, z_cond)
        
        # Apply classifier-free guidance
        if guidance_scale > 1.0:
            # Unconditional prediction (zero conditions)
            context_uncond = torch.zeros_like(context)
            z_cond_uncond = torch.zeros_like(z_cond)
            eps_uncond = self.model(z_t, t_emb, context_uncond, z_cond_uncond)
            
            # Guidance formula: eps = eps_uncond + scale * (eps_cond - eps_uncond)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond
        
        # Predict x0
        z0_est = (z_t - sqrt_one_minus * eps) / sqrt_alpha
        
        if t_idx == 0:
            return z0_est
        
        # Add noise for next step
        alpha_prev = self.alphas_cumprod[t_idx - 1].to(device)
        noise = torch.randn_like(z_t)
        z_prev = torch.sqrt(alpha_prev) * z0_est + torch.sqrt(1 - alpha_prev) * noise
        
        return z_prev

    # sample image-to-image
    # @torch.no_grad()
    # def sample(self, x_cond, cond_vec, guidance_scale, device):
    #     self.eval()
    #     B = x_cond.size(0)
    #     z_cond = self.get_latent(x_cond)
    #     z = torch.randn_like(z_cond)
    #     context = self.model.condition_proj(cond_vec)

    #     for i in tqdm(reversed(range(self.timesteps)), total=self.timesteps, desc="Sampling"):
    #         z = self.p_sample(z, z_cond, i, context, guidance_scale)

    #     x = self.vae_dec(z)
    #     x = (x.clamp(-1,1)+1)/2
    #     return x
    @torch.no_grad()
    def sample(self, x_cond, cond_vec, guidance_scale, device):
        """
        Generate images with classifier-free guidance.
        
        Args:
            x_cond: Conditioning images
            cond_vec: Conditioning vectors
            guidance_scale: Guidance strength (recommended: 3.0-7.0)
            device: Device to run on
        """
        self.eval()
        B = x_cond.size(0)
        
        # Get conditioning latent
        z_cond = self.get_latent(x_cond)
        
        # Start from random noise
        z = torch.randn_like(z_cond)
        
        # Project condition vector
        context = self.model.condition_proj(cond_vec)

        # Denoising loop
        for i in tqdm(reversed(range(self.timesteps)), total=self.timesteps, desc="Sampling"):
            z = self.p_sample(z, z_cond, i, context, guidance_scale)

        # Decode to image
        x = self.vae_dec(z)
        x = (x.clamp(-1, 1) + 1) / 2
        return x

    # @torch.no_grad()
    # def test_conditioning(self, x_cond1, x_cond2, cond_vec):
    #     z_cond1 = self.get_latent(x_cond1)
    #     z_cond2 = self.get_latent(x_cond2)
        
    #     z_t = torch.randn_like(z_cond1)
    #     t = torch.zeros(1, device=z_t.device, dtype=torch.long)
    #     t_emb = self.timestep_embedding(t, TIME_EMBED_DIM)
    #     context = self.model.condition_proj(cond_vec)
        
    #     eps1 = self.model(z_t, t_emb, context, z_cond1)
    #     eps2 = self.model(z_t, t_emb, context, z_cond2)
        
    #     diff = (eps1 - eps2).abs().mean()
    #     print(f"Prediction difference: {diff:.6f}")
    #     return diff
    @torch.no_grad()
    def test_conditioning(self, x_cond1, x_cond2, cond_vec):
        device = x_cond1.device
        z_cond1 = self.get_latent(x_cond1)
        z_cond2 = self.get_latent(x_cond2)
        z_t = torch.randn_like(z_cond1)
        t = torch.full((x_cond1.size(0),), self.timesteps // 2, device=device, dtype=torch.long)
        t_emb = self.timestep_embedding(t, TIME_EMBED_DIM)
        context = self.model.condition_proj(cond_vec)
        
        eps1 = self.model(z_t, t_emb, context, z_cond1)
        eps2 = self.model(z_t, t_emb, context, z_cond2)
        
        diff = (eps1 - eps2).abs().mean().item()
        z_cond_diff = (z_cond1 - z_cond2).abs().mean().item()
        
        print(f"  Latent condition difference (z_cond): {z_cond_diff:.6f}")
        print(f"  Prediction difference (eps): {diff:.6f}")
        print(f"  Ratio (eps_diff / z_cond_diff): {diff / (z_cond_diff + 1e-8):.4f}")
        
        return diff
# ============================================================
# TRAINING VAE
# ============================================================
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss: L1 tốt hơn MSE cho X-ray
    recon_loss = F.l1_loss(recon_x, x, reduction='mean')

    # KL divergence
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + 0.0005 * kld   


def train_vae(encoder, decoder, data_loader, epochs, lr, save_dir, device):
    encoder.to(device)
    decoder.to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        encoder.train()
        decoder.train()
        total_loss = 0.0
        pbar = tqdm(data_loader, desc=f"VAE Epoch {epoch}/{epochs}")

        for imgs, *_ in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            mu, logvar = encoder(imgs)
            logvar = torch.clamp(logvar, -20, 20)
            # Reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            recon_imgs = decoder(z)
            loss = vae_loss(recon_imgs, imgs, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(data_loader)
        print(f"[VAE Epoch {epoch}] Avg Loss: {avg_loss:.6f}")

        # Lưu model mỗi epoch
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict()
        }, os.path.join(save_dir, f"vae_epoch{epoch}.pt"))

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_diffusion_model(diffusion_model, data_loader, epochs, save_dir, device):
    diffusion_model.to(device)
    optimizer = torch.optim.AdamW(diffusion_model.model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,        # Reduce LR by half
        patience=5,        # Wait 5 epochs before reducing
        min_lr=1e-6     # Don't go below this
    )
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(1, epochs+1):
        diffusion_model.train()
        total_loss = 0.0
        pbar = tqdm(data_loader, desc=f"Epoch {ep}/{epochs}")

        for img_cond, img_target, cond_vec in pbar:
            img_cond = img_cond.to(device)
            img_target = img_target.to(device)
            cond_vec = cond_vec.to(device)

            optimizer.zero_grad()
            loss = diffusion_model(img_target, img_cond, cond_vec, 0.15)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(data_loader)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {ep}/{epochs}] Avg Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")

        # save checkpoint
        torch.save({
            'model_state_dict': diffusion_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_avg': avg_loss,
            'epoch': ep
        }, os.path.join(save_dir, f"epoch{ep}.pt"))


def resume_diffusion_training(diffusion_model, checkpoint_path, data_loader, total_epochs, save_dir, device):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model_chkps = torch.load(checkpoint_path, map_location=device)
    diffusion_model.load_state_dict(model_chkps['model_state_dict'])
    start_epoch = model_chkps['epoch']

    optimizer = torch.optim.AdamW(diffusion_model.model.parameters(), lr=LEARNING_RATE)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,      
    #     patience=5,        
    #     min_lr=1e-6     
    # )

    # Load lại optimizer state để training không bị "reset momentum"
    if 'optimizer' in model_chkps:
        optimizer.load_state_dict(model_chkps['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = 5e-5
        print(f"[INFO] Loaded optimizer state")
    else:
        print("[WARN] Optimizer checkpoint NOT FOUND → training will continue with fresh optimizer")

    print(f"[INFO] Resume training from epoch {start_epoch+1} → {total_epochs}")
    for ep in range(start_epoch+1, total_epochs+1):
        diffusion_model.train()
        total_loss = 0.0
        pbar = tqdm(data_loader, desc=f"Epoch {ep}/{total_epochs}")

        for img_cond, img_target, cond_vec in pbar:
            img_cond   = img_cond.to(device)
            img_target = img_target.to(device)
            cond_vec   = cond_vec.to(device)

            optimizer.zero_grad()
            loss = diffusion_model(img_target, img_cond, cond_vec)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(data_loader)
        # scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {ep}/{total_epochs}] Avg Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")

  

        # save model
        # model_path = os.path.join(save_dir, f"epoch{ep}.pt")
        # opt_path   = os.path.join(save_dir, f"epoch{ep}_optimizer.pt")

        # torch.save(diffusion_model.state_dict(), model_path)
        # torch.save(optimizer.state_dict(), opt_path)

        torch.save({
            'model_state_dict': diffusion_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_avg': avg_loss,
            'epoch': ep
        }, os.path.join(save_dir, f"epoch{ep}.pt"))



# ============================================================
# TEST / SAMPLE FUNCTION
# ============================================================
def test_diffusion_model(diffusion_model, checkpoint_path, x_cond, cond_vec, guidance_scale=3.0, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    diffusion_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    diffusion_model.to(device)
    diffusion_model.eval()

    x_cond = x_cond.to(device)
    cond_vec = cond_vec.to(device)

    with torch.no_grad():
        generated = diffusion_model.sample(x_cond, cond_vec, guidance_scale, device)
    return generated

def test_diffusion_model2(diffusion_model, checkpoint_path, x_cond, cond_vec, guidance_scale=3.0, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model_chkps = torch.load(checkpoint_path, map_location=device)
    diffusion_model.load_state_dict(model_chkps['model_state_dict'])
    diffusion_model.to(device)
    diffusion_model.eval()

    x_cond = x_cond.to(device)
    cond_vec = cond_vec.to(device)

    with torch.no_grad():
        generated = diffusion_model.sample(x_cond, cond_vec, guidance_scale, device)
    return generated

def test_vae(encoder, decoder, imgs, device):
    encoder.to(device)
    decoder.to(device)

    encoder.eval()
    decoder.eval()

    imgs = imgs.to(device)

    with torch.no_grad():
        mu, logvar = encoder(imgs)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon_imgs = decoder(z)

    return recon_imgs

import matplotlib.pyplot as plt
import torch

def show_vae_results(imgs, recon_imgs, num=8):
    """
    imgs: ảnh gốc
    recon_imgs: ảnh tái tạo từ decoder
    num: số ảnh muốn hiển thị
    """

    # Lấy tối đa num ảnh để hiển thị
    imgs = imgs[:num].cpu()
    recon_imgs = recon_imgs[:num].cpu()

    # Nếu dữ liệu ở [-1,1] -> đưa về [0,1] để hiển thị
    imgs = (imgs + 1) / 2
    recon_imgs = (recon_imgs + 1) / 2

    plt.figure(figsize=(num * 2, 4))

    for i in range(num):
        # Ảnh gốc
        plt.subplot(2, num, i + 1)
        plt.imshow(imgs[i].permute(1, 2, 0), cmap="gray")
        plt.axis("off")
        plt.title("Original")

        # Ảnh tái tạo
        plt.subplot(2, num, num + i + 1)
        plt.imshow(recon_imgs[i].permute(1, 2, 0), cmap="gray")
        plt.axis("off")
        plt.title("Reconstructed")

    plt.tight_layout()
    plt.show()

# ============================================================
# MAIN
# ============================================================
from data import load_data, XrayPairedDataset  
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_DIR_VAE = "vae_chkps"
    SAVE_DIR = "diffusion_chkps_2"
    # LOAD DATA
    data_loader = load_data() 
    # TRAIN VAE
    def trainvae(save_dir=SAVE_DIR_VAE):
        encoder = VAEEncoder()
        decoder = VAEDecoder()
        train_vae(encoder, decoder, data_loader, epochs=20, lr=1e-4, save_dir=save_dir, device=device)
    def testvae(save_dir=SAVE_DIR_VAE):
        vae_enc = VAEEncoder()
        vae_dec = VAEDecoder()

        vae_ckpt = torch.load(f"{save_dir}/vae_epoch20.pt", map_location=device)

        vae_enc.load_state_dict(vae_ckpt['encoder'])
        vae_dec.load_state_dict(vae_ckpt['decoder'])

        for p in vae_enc.parameters():
                p.requires_grad = False
        for p in vae_dec.parameters():
                p.requires_grad = False

        sample_batch = next(iter(data_loader))
        img_cond, img_target, cond_vec = sample_batch
        recon_imgs = test_vae(vae_enc, vae_dec, img_cond, device)
        show_vae_results(img_cond, recon_imgs, num=8)
        # with torch.no_grad():
        #     z = vae_enc(img_cond)[0]
        # print("latent shape:", z.shape)

    # CREATE MODEL
    def create_model(num_epoch=20, save_dir=SAVE_DIR_VAE):
        vae_enc = VAEEncoder()
        vae_dec = VAEDecoder()

        vae_ckpt = torch.load(f"{save_dir}/vae_epoch{num_epoch}.pt", map_location=device)

        vae_enc.load_state_dict(vae_ckpt['encoder'])
        vae_dec.load_state_dict(vae_ckpt['decoder'])

        for p in vae_enc.parameters():
            p.requires_grad = False
        for p in vae_dec.parameters():
            p.requires_grad = False
        vae_enc.eval()
        vae_dec.eval()

        unet = ConditionalUNet().to(device)

        # unet = ConditionalUNet()
        # total_params = sum(p.numel() for p in unet.parameters())
        # print(f"UNet parameters: {total_params:,}")

        diffusion_model = GaussianDiffusion(unet, vae_enc, vae_dec, NUM_TIMESTEPS)

        return diffusion_model
    
    # TRAIN
    def train(save_dir=SAVE_DIR):
        diffusion_model = create_model()
        train_diffusion_model(diffusion_model, data_loader, epochs=50, save_dir=save_dir, device=device)

    # TEST
    def test(num_epoch, save_dir=SAVE_DIR):
        diffusion_model= create_model()
        sample_batch = next(iter(data_loader))
        img_cond, img_target, cond_vec = sample_batch
        generated_imgs = test_diffusion_model2(diffusion_model, f"{save_dir}/epoch{num_epoch}.pt", img_cond, cond_vec, guidance_scale=5, device=device)

        
        # SHOW RESULTS
        num_imgs = img_cond.size(0)
        fig, axes = plt.subplots(2, num_imgs, figsize=(num_imgs*4, 8))
        for i in range(num_imgs):
            axes[0,i].imshow(img_cond[i,0].cpu(), cmap="gray")
            axes[0,i].set_title("Input")
            axes[0,i].axis("off")
            axes[1,i].imshow(generated_imgs[i,0].cpu(), cmap="gray")
            axes[1,i].set_title("Generated")
            axes[1,i].axis("off")
        plt.show()
    
    def resume_train(to_epoch, checkpoint_path, save_dir=SAVE_DIR):
        model = create_model().to(device)

        resume_diffusion_training(
            diffusion_model=model,
            checkpoint_path=checkpoint_path,
            data_loader=data_loader,
            total_epochs=to_epoch,   
            save_dir=save_dir,
            device=device
        )

    import argparse
    parser = argparse.ArgumentParser(description="Mode")
    parser.add_argument('mode', type=str, help="Mode selected")
    args = parser.parse_args()
    if args.mode == "train":
        train()
    elif args.mode == "test":
        # testvae()
        test(num_epoch=122)
    elif args.mode == "train_vae":
        trainvae()
    elif args.mode == "test_vae":
        testvae()
    elif args.mode == "train_resume":
        from_epoch = 122
        to_epoch = 170
        resume_train(to_epoch,checkpoint_path=f"{SAVE_DIR}/epoch{from_epoch}.pt")
    ##################
    def test_model_conditioning(diffusion_model, checkpoint_path, x_cond1, x_cond2, cond_vec, device="cuda"):
        """
        Test if the diffusion model properly uses conditioning information.
        
        Args:
            diffusion_model: Your GaussianDiffusion model
            checkpoint_path: Path to model checkpoint
            x_cond1: First conditioning image (B, C, H, W)
            x_cond2: Second conditioning image (B, C, H, W)
            cond_vec: Conditioning vector (B, D)
            device: Device to run on
        
        Returns:
            diff: Difference metric (float) - larger means better conditioning
        """
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = torch.load(checkpoint_path, map_location=device)
        diffusion_model.load_state_dict(model['modelload_state_dict'])
        diffusion_model.to(device)
        diffusion_model.eval()

        x_cond1 = x_cond1.to(device)
        x_cond2 = x_cond2.to(device)
        cond_vec = cond_vec.to(device)

        with torch.no_grad():
            diff = diffusion_model.test_conditioning(x_cond1, x_cond2, cond_vec)
        
        # Interpret results
        print(f"\n{'='*60}")
        print(f"Conditioning Test Results:")
        print(f"{'='*60}")
        print(f"Prediction difference: {diff:.6f}")
        
        if diff < 0.01:
            print("❌ WARNING: Very small difference - model is likely IGNORING conditioning!")
            print("   → This explains why different inputs produce identical outputs")
        elif diff < 0.1:
            print("⚠️  CAUTION: Small difference - model uses conditioning weakly")
            print("   → Model may produce similar outputs for different inputs")
        else:
            print("✅ GOOD: Significant difference - model is using conditioning properly")
        print(f"{'='*60}\n")
        
        return diff


    # from torchvision import transforms
    # from PIL import Image

    # diffusion = create_model()
    
    # # Load two different input images
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5], [0.5])
    # ])
    
    # img1 = transform(Image.open("images_1/00000001_000.png")).unsqueeze(0)
    # img2 = transform(Image.open("images_1/00000005_000.png")).unsqueeze(0)
    
    # # Conditioning vector (adjust based on your setup)
    # sample_batch = next(iter(data_loader))
    # img_cond, img_target, cond_vec = sample_batch
    
    # # Test conditioning
    # diff = test_model_conditioning(
    #     diffusion_model=diffusion,
    #     checkpoint_path=f"{SAVE_DIR}/epoch50.pt",
    #     x_cond1=img1,
    #     x_cond2=img2,
    #     cond_vec=cond_vec,
    #     device="cuda"
    # )
    def test_single_input_multiple_conditions(num_epoch, save_dir=SAVE_DIR, specific_diseases=None):
        """
        Test với 1 ảnh input và 5 điều kiện bệnh
        
        Args:
            num_epoch: số epoch của checkpoint
            save_dir: thư mục lưu model
            specific_diseases: danh sách bệnh cụ thể (None để chọn ngẫu nhiên)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diffusion_model = create_model()
        
        # Load model
        model_chkps = torch.load(f"{save_dir}/epoch{num_epoch}.pt", map_location=device)
        diffusion_model.load_state_dict(model_chkps['model_state_dict'])
        diffusion_model.to(device)
        diffusion_model.eval()
        
        # Load dataset
       
        dataset = XrayPairedDataset()
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Lấy 1 ảnh input
        img_cond, _, _ = next(iter(data_loader))
        img_cond = img_cond.to(device)
        
        # Chuẩn bị conditions
        all_diseases = dataset.label_names
        
        if specific_diseases is not None:
            # Sử dụng các bệnh được chỉ định
            condition_names = specific_diseases[:5]  # Lấy tối đa 5 bệnh
            condition_vectors = []
            
            for disease_name in condition_names:
                if disease_name in all_diseases:
                    disease_idx = all_diseases.index(disease_name)
                    cond_vec = torch.zeros(len(all_diseases), dtype=torch.float32)
                    cond_vec[disease_idx] = 1.0
                    condition_vectors.append(cond_vec)
                else:
                    print(f"Warning: Bệnh '{disease_name}' không có trong dataset")
            
            # Nếu không đủ 5 bệnh, thêm ngẫu nhiên
            while len(condition_vectors) < 5:
                disease_idx = torch.randint(0, len(all_diseases), (1,)).item()
                disease_name = all_diseases[disease_idx]
                if disease_name not in condition_names:
                    cond_vec = torch.zeros(len(all_diseases), dtype=torch.float32)
                    cond_vec[disease_idx] = 1.0
                    condition_vectors.append(cond_vec)
                    condition_names.append(disease_name)
        else:
            # Chọn ngẫu nhiên 5 bệnh
            selected_indices = torch.randperm(len(all_diseases))[:5]
            condition_vectors = []
            condition_names = []
            
            for idx in selected_indices:
                disease_name = all_diseases[idx]
                cond_vec = torch.zeros(len(all_diseases), dtype=torch.float32)
                cond_vec[idx] = 1.0
                condition_vectors.append(cond_vec)
                condition_names.append(disease_name)
        
        condition_vectors = torch.stack(condition_vectors).to(device)
        
        # Generate ảnh
        generated_imgs = []
        with torch.no_grad():
            for i in range(5):
                cond_vec = condition_vectors[i:i+1]
                generated = diffusion_model.sample(img_cond, cond_vec, guidance_scale=5.0, device=device)
                generated_imgs.append(generated[0])
        
        # Hiển thị kết quả
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        
        # Ảnh input
        axes[0, 0].imshow(img_cond[0, 0].cpu(), cmap="gray")
        axes[0, 0].set_title("Input Image\n(No Finding)", fontsize=12)
        axes[0, 0].axis('off')
        
        axes[1, 0].axis('off')  # Ô trống dưới input
        
        # Các ảnh generated
        for i in range(5):
            axes[0, i+1].imshow(generated_imgs[i][0].cpu(), cmap="gray")
            axes[0, i+1].set_title(f"Generated: {condition_names[i]}", fontsize=10)
            axes[0, i+1].axis('off')
            
            # axes[1, i+1].text(0.5, 0.5, condition_names[i], 
            #                 horizontalalignment='center', verticalalignment='center',
            #                 transform=axes[1, i+1].transAxes, fontsize=12, weight='bold')
            axes[1, i+1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return generated_imgs, condition_names


    # Test với 1 input và 5 conditions ngẫu nhiên
    test_single_input_multiple_conditions(122)
    
    # Test với các bệnh cụ thể
    # test_single_input_multiple_conditions(100, specific_diseases=[
    #     'Atelectasis', 'Effusion', 'Pneumonia', 'Cardiomegaly', 'Infiltration'
    # ])
    
    # Test với nhiều inputs
  



