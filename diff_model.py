import math
import random
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure reproducibility
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device, dtype=torch.float32)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])

    return torch.clip(betas, 0.0001, 0.9999)

timesteps = 1000

betas = cosine_beta_schedule(timesteps).to(device)
alphas = 1. - betas

alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.ones(1, device=device, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]])

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_log_variance = torch.log(torch.clamp(posterior_variance, min=1e-20))

def q_sample(x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    sqrt_alpha_bar = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    factor = math.log(10000) / max(half_dim - 1, 1)

    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -factor)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # odd dims
        emb = F.pad(emb, (0,1))

    return emb

# CNN Diffusion Architecture

class TimeBias(nn.Module):
    def __init__(self, time_emb_dim, channels):
        super().__init__()
        self.proj = nn.Linear(time_emb_dim, channels)
    def forward(self, h, t_emb):
        # h: [B,C,H,W], t_emb: [B, time_emb_dim]
        return h + self.proj(t_emb).unsqueeze(-1).unsqueeze(-1)
class DiffCNN(nn.Module):
    def __init__(self, in_channels=3, cond_channels=3, time_emb_dim=256):
        super().__init__()
        cin = in_channels + cond_channels

        # timestep embedding MLP (use your get_timestep_embedding to feed this)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim*2), nn.SiLU(),
            nn.Linear(time_emb_dim*2, time_emb_dim)
        )

        # encoder
        self.conv1 = nn.Conv2d(cin, 64, kernel_size=5, padding=2, bias=True)
        self.t1    = TimeBias(time_emb_dim, 64)
        self.act1  = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True)
        self.t2    = TimeBias(time_emb_dim, 64)
        self.act2  = nn.SiLU(inplace=True)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True)
        self.t3    = TimeBias(time_emb_dim, 64)
        self.act3  = nn.SiLU(inplace=True)

        # decoder
        self.deconv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)
        self.t4      = TimeBias(time_emb_dim, 32)
        self.act4    = nn.SiLU(inplace=True)

        # predict noise ε (no clamp, no global residual add)
        self.out_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=True)

    def forward(self, x_t, t, cond):
        # t: LongTensor [B]
        t_emb = get_timestep_embedding(t, self.time_mlp[0].in_features)  # -> [B, time_emb_dim]
        t_emb = self.time_mlp(t_emb)                                     # -> [B, time_emb_dim]

        x = torch.cat([x_t, cond], dim=1)  # [B, in+cond, H, W]

        x = self.conv1(x); x = self.t1(x, t_emb); x = self.act1(x)
        x = self.conv2(x); x = self.t2(x, t_emb); x = self.act2(x)
        x = self.conv3(x); x = self.t3(x, t_emb); x = self.act3(x)

        x = self.deconv1(x); x = self.t4(x, t_emb); x = self.act4(x)
        eps_hat = self.out_conv(x)
        return eps_hat

# Unet Diffusion Architecture

def Norm(kind, c):
    if kind == 'group':    return nn.GroupNorm(32 if c >= 32 else 1, c)
    if kind == 'instance': return nn.InstanceNorm2d(c, affine=True)
    if kind == 'batch':    return nn.BatchNorm2d(c)
    raise ValueError(f"Unknown norm: {kind}")
class FiLM(nn.Module):
    def __init__(self, time_emb_dim, channels):
        super().__init__()
        self.to_gamma_beta = nn.Linear(time_emb_dim, 2 * channels)

    def forward(self, h, t_emb):
        gb = self.to_gamma_beta(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B,2C,1,1]
        gamma, beta = gb.chunk(2, dim=1)                            # [B,C,1,1]

        return (1 + gamma) * h + beta
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_dropout=False, norm='group', act='silu'):
        super().__init__()
        Act = nn.SiLU if act == 'silu' else nn.ReLU

        # stage 1
        self.n1   = Norm(norm, in_ch)
        self.f1   = FiLM(time_emb_dim, in_ch)
        self.act1 = Act(inplace=True)
        self.c1   = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)

        # stage 2
        self.n2   = Norm(norm, out_ch)
        self.f2   = FiLM(time_emb_dim, out_ch)
        self.act2 = Act(inplace=True)
        self.drop = nn.Dropout2d(0.2) if use_dropout else nn.Identity()
        self.c2   = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

        # shortcut
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

        # stable init: start near identity
        nn.init.zeros_(self.c2.weight)

    def forward(self, x, t_emb):
        h = self.c1(self.act1(self.f1(self.n1(x), t_emb)))
        h = self.c2(self.drop(self.act2(self.f2(self.n2(h), t_emb))))

        return h + self.proj(x)

class SelfAttention(nn.Module):
    def __init__(self, in_ch, norm='group'):
        super().__init__()
        self.norm = Norm(norm, in_ch)
        d = max(1, in_ch // 8)
        self.q = nn.Conv2d(in_ch, d, 1, bias=False)
        self.k = nn.Conv2d(in_ch, d, 1, bias=False)
        self.v = nn.Conv2d(in_ch, in_ch, 1, bias=False)

        self.proj = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x)
        q = self.q(h).view(B, -1, H*W).transpose(1, 2)     # [B, HW, d]
        k = self.k(h).view(B, -1, H*W)                     # [B, d,  HW]
        v = self.v(h).view(B,  C, H*W).transpose(1, 2)     # [B, HW, C]

        attn = torch.softmax((q.float() @ k.float()) * (q.size(-1) ** -0.5), dim=-1).to(v.dtype)  # [B,HW,HW]
        out = (attn @ v).transpose(1, 2).view(B, C, H, W)

        return x + self.gamma * self.proj(out)
class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_dropout=False, norm='group', act='silu'):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch, time_emb_dim, use_dropout, norm, act)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x, t_emb):
        feat = self.block(x, t_emb)   # skip
        down = self.pool(feat)        # next scale

        return feat, down
class Decoder(nn.Module):
    def __init__(self, up_in, skip_in, out_ch, time_emb_dim, use_dropout=False, norm='group', act='silu'):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(up_in, out_ch, 3, padding=1, bias=False)
        )
        self.block = ResidualBlock(out_ch + skip_in, out_ch, time_emb_dim, use_dropout, norm, act)

    def forward(self, x_deep, x_skip, t_emb):
        x = self.up(x_deep)
        if x.size()[2:] != x_skip.size()[2:]:
            x = F.interpolate(x, size=x_skip.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x_skip, x], dim=1)

        return self.block(x, t_emb)
class DiffUNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=3, out_channels=3,
                 time_emb_dim=256, use_dropout=False, norm='group', act='silu'):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim*4), nn.SiLU(),
            nn.Linear(time_emb_dim*4, time_emb_dim)
        )

        C_in = in_channels + cond_channels  # concat [x_t, blur]

        self.in_block = ResidualBlock(C_in, 64, time_emb_dim, use_dropout, norm, act)

        self.enc1 = Encoder(64,   128, time_emb_dim, use_dropout, norm, act)
        self.enc2 = Encoder(128,  256, time_emb_dim, use_dropout, norm, act)
        self.enc3 = Encoder(256,  512, time_emb_dim, use_dropout, norm, act)
        self.enc4 = Encoder(512, 1024, time_emb_dim, use_dropout, norm, act)

        # bottleneck @ 16x16
        self.mid1 = ResidualBlock(1024, 1024, time_emb_dim, use_dropout, norm, act)
        self.attn = SelfAttention(1024, norm=norm)  # attention only at low res
        self.mid2 = ResidualBlock(1024, 1024, time_emb_dim, use_dropout, norm, act)

        # decode
        self.dec1 = Decoder(1024, 512, 512, time_emb_dim, use_dropout, norm, act)
        self.dec2 = Decoder(512,  256, 256, time_emb_dim, use_dropout, norm, act)
        self.dec3 = Decoder(256,  128, 128, time_emb_dim, use_dropout, norm, act)
        self.dec4 = Decoder(128,   64,  64,  time_emb_dim, use_dropout, norm, act)

        self.out_conv = nn.Conv2d(64, out_channels, 1)  # predicts ε (noise)

    def forward(self, x_t, t, cond):
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        x = torch.cat([x_t, cond], dim=1)   # [B, 6, H, W]
        x1 = self.in_block(x, t_emb)        # 64

        f2, x2 = self.enc1(x1, t_emb)       # 128
        f3, x3 = self.enc2(x2, t_emb)       # 256
        f4, x4 = self.enc3(x3, t_emb)       # 512
        f5, x5 = self.enc4(x4, t_emb)       # 1024

        m = self.mid1(x5, t_emb)
        m = self.attn(m)
        m = self.mid2(m, t_emb)

        x = self.dec1(m,  f4, t_emb)        # 512
        x = self.dec2(x,  f3, t_emb)        # 256
        x = self.dec3(x,  f2, t_emb)        # 128
        x = self.dec4(x,  x1, t_emb)        # 64

        return self.out_conv(x)             # ε̂

# ResNet Diffusion Architecture
class ResBlock(nn.Module):
    def __init__(self, c, time_emb_dim, norm='group', act='silu'):
        super().__init__()
        Act = nn.ReLU if act == 'relu' else nn.SiLU

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(c, c, 3, bias=False)
        self.n1 = Norm(norm, c)
        self.f1 = FiLM(time_emb_dim, c)
        self.act1 = Act(inplace=True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(c, c, 3, bias=False)
        self.n2 = Norm(norm, c)
        self.f2 = FiLM(time_emb_dim, c)

        # stable: start near identity
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x, t_emb):
        h = self.pad1(x); h = self.conv1(h); h = self.n1(h); h = self.f1(h, t_emb); h = self.act1(h)
        h = self.pad2(h); h = self.conv2(h); h = self.n2(h);  h = self.f2(h, t_emb)

        return x + h
class FuseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, norm='group', act='silu'):
        super().__init__()
        Act = nn.ReLU if act == 'relu' else nn.SiLU
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm = Norm(norm, out_ch)
        self.film = FiLM(time_emb_dim, out_ch)
        self.act  = Act(inplace=True)

    def forward(self, x, t_emb):
        return self.act(self.film(self.norm(self.conv(x)), t_emb))
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='group', time_emb_dim=256, act='silu'):
        super().__init__()
        Act = nn.ReLU if act == 'relu' else nn.SiLU
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.n = Norm(norm, out_ch)
        self.film = FiLM(time_emb_dim, out_ch)
        self.act = Act(inplace=True)

    def forward(self, x, t_emb):
        h = self.conv(x); h = self.n(h); h = self.film(h, t_emb); return self.act(h)
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm='group', time_emb_dim=256, act='silu'):
        super().__init__()
        Act = nn.ReLU if act == 'relu' else nn.SiLU
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.n = Norm(norm, out_ch)
        self.film = FiLM(time_emb_dim, out_ch)
        self.act = Act(inplace=True)

    def forward(self, x, t_emb):
        h = self.up(x); h = self.conv(h); h = self.n(h); h = self.film(h, t_emb); return self.act(h)
class DiffResNet(nn.Module):
    def __init__(self, in_channels=3, cond_channels=3, out_channels=3,
                 num_resnet_blocks=9, norm='group', time_emb_dim=256, act='silu'):
        super().__init__()
        Act = nn.ReLU if act == 'relu' else nn.SiLU
        self.time_emb_dim = time_emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim*4), Act(),
            nn.Linear(time_emb_dim*4, time_emb_dim)
        )

        C_in = in_channels + cond_channels  # concat [x_t, condition]

        # stem
        self.stem_pad = nn.ReflectionPad2d(3)
        self.stem_conv = nn.Conv2d(C_in, 64, kernel_size=7, bias=False)
        self.stem_norm = Norm(norm, 64)
        self.stem_film = FiLM(time_emb_dim, 64)
        self.stem_act  = Act(inplace=True)

        # down: 64->128->256
        self.down1 = DownBlock(64, 128, norm=norm, time_emb_dim=time_emb_dim, act=act)
        self.down2 = DownBlock(128, 256, norm=norm, time_emb_dim=time_emb_dim, act=act)

        # 9 residual blocks @256
        self.blocks = nn.ModuleList([ResBlock(256, time_emb_dim, norm=norm, act=act)
                                     for _ in range(num_resnet_blocks)])

        # up: 256->128->64
        self.up1 = UpBlock(256, 128, norm=norm, time_emb_dim=time_emb_dim, act=act)
        self.up2 = UpBlock(128,  64,  norm=norm, time_emb_dim=time_emb_dim, act=act)

        # long-skip fuse blocks (concat + fuse)
        self.fuse1 = FuseBlock(in_ch=128+128, out_ch=128, time_emb_dim=time_emb_dim, norm=norm, act=act)  # 128×128 skip
        self.fuse0 = FuseBlock(in_ch= 64+ 64, out_ch= 64, time_emb_dim=time_emb_dim, norm=norm, act=act)  # 256×256 skip

        # output
        self.out_pad = nn.ReflectionPad2d(3)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=7, bias=True)

    @staticmethod
    def _match_hw(x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, x_t, t, cond):
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        x = torch.cat([x_t, cond], dim=1)

        # stem (256×256, 64ch)
        h = self.stem_pad(x); h = self.stem_conv(h); h = self.stem_norm(h); h = self.stem_film(h, t_emb); h = self.stem_act(h)
        s0 = h.clone()

        # down path
        h = self.down1(h, t_emb)  # -> 128×128, 128ch
        s1 = h.clone()
        h = self.down2(h, t_emb)  # -> 64×64, 256ch

        # bottleneck resblocks
        for blk in self.blocks:
            h = blk(h, t_emb)

        # up1: 64×64->128×128 (128ch)
        h = self.up1(h, t_emb)
        s1 = self._match_hw(s1, h)
        h = torch.cat([s1, h], dim=1)
        h = self.fuse1(h, t_emb)

        # up2: 128×128->256×256 (64ch)
        h = self.up2(h, t_emb)
        s0 = self._match_hw(s0, h)
        h = torch.cat([s0, h], dim=1)
        h = self.fuse0(h, t_emb)

        h = self.out_pad(h)
        return self.out_conv(h)  # ε̂
# EMA
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}

    def copy_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data = self.shadow[name].clone()


@torch.no_grad()
def sample_ddpm(model, condition, scheduler_params, device, num_steps=None):
    betas  = scheduler_params['betas'].to(device)
    alphas = 1.0 - betas
    abar   = torch.cumprod(alphas, dim=0)
    T      = betas.numel()

    B, C, H, W = condition.shape
    x = torch.randn((B, C, H, W), device=device)
    eps = 1e-5
    model.eval()

    if num_steps is None or num_steps >= T:
        ts = torch.arange(T, device=device, dtype=torch.long)
    else:
        # stride schedule that includes 0 and T-1, no duplicates
        step = max(1, (T - 1) // max(1, num_steps - 1))
        ts = torch.arange(0, T, step, device=device, dtype=torch.long)
        if ts[-1].item() != T - 1: ts = torch.cat([ts, torch.tensor([T - 1], device=device)])
        if ts[0].item()  != 0:     ts = torch.cat([torch.tensor([0], device=device), ts])
        ts = torch.unique(ts, sorted=True)

    for i in range(ts.numel() - 1, -1, -1):
        t_idx = ts[i].item()
        t_prev_idx = ts[i - 1].item() if i > 0 else -1

        abar_t    = abar[t_idx]
        abar_prev = (abar[t_prev_idx] if t_prev_idx >= 0
                     else torch.tensor(1.0, device=device, dtype=abar.dtype))

        # predict noise at the ORIGINAL t
        t_b = torch.full((B,), t_idx, device=device, dtype=torch.long)
        eps_pred = model(x, t_b, condition)

        # reconstruct x0 and clamp
        x0_pred = (x - torch.sqrt(1.0 - abar_t) * eps_pred) / torch.sqrt(abar_t + eps)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        if i > 0:
            alpha_eff = (abar_t / (abar_prev + eps)).clamp(min=eps, max=1.0)  # in (0,1]
            beta_eff  = 1.0 - alpha_eff
            post_var  = torch.clamp((1.0 - abar_prev) / (1.0 - abar_t + eps) * beta_eff, min=eps)

            mean = (torch.sqrt(abar_prev) * x0_pred
                    + torch.sqrt(1.0 - abar_prev - post_var) * eps_pred)

            x = mean + torch.sqrt(post_var) * torch.randn_like(x)
        else:
            mean = (torch.sqrt(abar_prev) * x0_pred
                    + torch.sqrt(1.0 - abar_prev) * eps_pred)
            x = mean

    return x
@torch.no_grad()
def sample_ddim(model, condition, scheduler_params, device, num_steps, eta=0.0):
    betas = scheduler_params['betas'].to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    T = len(betas)

    ddim_timesteps = torch.linspace(0, T-1, steps=num_steps, device=device).long().flip(0)
    ddim_timesteps = torch.unique_consecutive(ddim_timesteps)

    B, C, H, W = condition.shape
    x_t = torch.randn((B, C, H, W), device=device)

    model.eval()
    for i, t in enumerate(ddim_timesteps):
        t_scalar = t.item()
        t_batch = torch.full((B,), t_scalar, device=device, dtype=torch.long)

        pred_noise = model(x_t, t_batch, condition)
        alpha_t = alphas_cumprod[t_scalar]

        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        x0_pred = x0_pred.clamp(-1, 1)

        if i == len(ddim_timesteps) - 1:
            x_t = x0_pred
        else:
            t_next = int(ddim_timesteps[i + 1].item())
            alpha_next = alphas_cumprod[t_next]

            sigma = (eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_next)).clamp(min=0.0)
            noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
            coeff_eps = torch.sqrt(torch.clamp(1 - alpha_next - sigma**2, min=0.0))

            x_t = torch.sqrt(alpha_next) * x0_pred + coeff_eps * pred_noise + sigma * noise

    return x_t
