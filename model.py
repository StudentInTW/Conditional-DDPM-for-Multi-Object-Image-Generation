import math
import torch
import torch.nn as nn
import torch.nn.functional as F





class SinusoidalTimeEmbedding(nn.Module):
  """
    Convert timestep t into sinusoidal embedding.

    Input:
        t: [B]

    Output:
        emb: [B, dim]
    """
    # This is so smart!
  def __init__(self,dim):
    super().__init__()
    self.dim = dim
    
  def forward(self,t):
    
    device = t.device
    # Since machine can't feel the difference between t= 500 and t=501
    # We use a mathematical way to express time.
    # We use "phase", namely  sin and cos to express a clock to tell time  
    #  let say we have 256 dim. We use sin and cos to express these 256 dim, each 128.
    half_dim = self.dim //2
    
    # We want the fasteset (second hand) to be 10000 times slower than hour hand. for each hand, they separate the speed evenly. So the final hand is 10000 times slower.
    emb_scale = math.log(10000) / (half_dim-1)
    # [0, 1, 2, 3, ..., 127] * -0.0725
    # with exp =>freq = [1.0, 0.93, 0.86, ..., 0.0001]
    freq = torch.exp(torch.arange(half_dim,device = device )* -emb_scale )


    # We now got the time and speed of hands
    # t: [B] -> [B, 1]
    t = t.float().unsqueeze(1)

    # [B, half_dim]
    emb = t * freq.unsqueeze(0)

    # [B, dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    return emb


class ResidualBlock(nn.Module):
    """
    ResBlock with time + condition embedding injection.

    Input:
        x:   [B, in_channels, H, W]
        emb: [B, emb_dim]

    Output:
        h:   [B, out_channels, H, W]
    """
    def __init__(self, in_channels, out_channels, emb_dim, num_groups=8):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.emb_proj = nn.Linear(emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))

        # emb: [B, emb_dim] -> [B, out_channels, 1, 1]
        emb_out = self.emb_proj(F.silu(emb))
        emb_out = emb_out[:, :, None, None]

        h = h + emb_out
        h = self.conv2(F.silu(self.norm2(h)))

        return h + self.shortcut(x)

    


class Downsample(nn.Module):
    """
    Downsample spatial size by 2.

    [B, C, H, W] -> [B, C, H/2, W/2]
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsample spatial size by 2.

    [B, C, H, W] -> [B, C, 2H, 2W]
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.conv(x)



class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for DDPM noise prediction.

    Input:
        x:      noisy image x_t, [B, 3, 64, 64]
        t:      timestep,       [B]
        labels: multi-hot label, [B, 24]

    Output:
        predicted noise,        [B, 3, 64, 64]
    """
    def __init__(
        self,
        image_channels=3,
        num_classes=24,
        base_channels=64,
        emb_dim=256
    ):
        super().__init__()

        self.image_channels = image_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.emb_dim = emb_dim

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # Multi-label condition embedding
        self.label_embedding = nn.Sequential(
            nn.Linear(num_classes, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )


        # Initial convolution
        self.init_conv = nn.Conv2d(
            image_channels,
            base_channels,
            kernel_size=3,
            padding=1
        )

        # Down path
        self.down_res1 = ResidualBlock(
            base_channels,
            base_channels,
            emb_dim
        )
        self.downsample1 = Downsample(base_channels)

        self.down_res2 = ResidualBlock(
            base_channels,
            base_channels * 2,
            emb_dim
        )
        self.downsample2 = Downsample(base_channels * 2)

        self.down_res3 = ResidualBlock(
            base_channels * 2,
            base_channels * 4,
            emb_dim
        )
        self.downsample3 = Downsample(base_channels * 4)

        # Middle
        self.mid_res1 = ResidualBlock(
            base_channels * 4,
            base_channels * 4,
            emb_dim
        )
        self.mid_res2 = ResidualBlock(
            base_channels * 4,
            base_channels * 4,
            emb_dim
        )

        # Up path
        self.upsample3 = Upsample(base_channels * 4)
        self.up_res3 = ResidualBlock(
            base_channels * 8,
            base_channels * 2,
            emb_dim
        )

        self.upsample2 = Upsample(base_channels * 2)
        self.up_res2 = ResidualBlock(
            base_channels * 4,
            base_channels,
            emb_dim
        )

        self.upsample1 = Upsample(base_channels)
        self.up_res1 = ResidualBlock(
            base_channels * 2,
            base_channels,
            emb_dim
        )


         # Output
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(
            base_channels,
            image_channels,
            kernel_size=3,
            padding=1
        )


    def forward(self, x, t, labels):
        """
        x:      [B, 3, 64, 64]
        t:      [B]
        labels: [B, 24]
        """
        labels = labels.float()

        time_emb = self.time_embedding(t)
        label_emb = self.label_embedding(labels)

        emb = time_emb + label_emb

        # Initial
        x = self.init_conv(x)

        # Down path
        h1 = self.down_res1(x, emb)              # [B, C, 64, 64]
        x = self.downsample1(h1)                 # [B, C, 32, 32]

        h2 = self.down_res2(x, emb)              # [B, 2C, 32, 32]
        x = self.downsample2(h2)                 # [B, 2C, 16, 16]

        h3 = self.down_res3(x, emb)              # [B, 4C, 16, 16]
        x = self.downsample3(h3)                 # [B, 4C, 8, 8]

        # Middle
        x = self.mid_res1(x, emb)                # [B, 4C, 8, 8]
        x = self.mid_res2(x, emb)                # [B, 4C, 8, 8]

        # Up path
        x = self.upsample3(x)                    # [B, 4C, 16, 16]
        x = torch.cat([x, h3], dim=1)            # [B, 8C, 16, 16]
        x = self.up_res3(x, emb)                 # [B, 2C, 16, 16]

        x = self.upsample2(x)                    # [B, 2C, 32, 32]
        x = torch.cat([x, h2], dim=1)            # [B, 4C, 32, 32]
        x = self.up_res2(x, emb)                 # [B, C, 32, 32]

        x = self.upsample1(x)                    # [B, C, 64, 64]
        x = torch.cat([x, h1], dim=1)            # [B, 2C, 64, 64]
        x = self.up_res1(x, emb)                 # [B, C, 64, 64]

        # Output predicted noise
        x = self.out_conv(F.silu(self.out_norm(x)))

        return x





