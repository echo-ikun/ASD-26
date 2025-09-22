import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryDetectionNet(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=128, out_dim=128, kernel_size=5):
        """
        Args:
            in_dim: input feature dimension (default=128)
            hidden_dim: convolution channel number for boundary detection (default=128)
            out_dim: output boundary feature dimension (default=128)
            kernel_size: Conv1D window size (3 or 5)
        """
        super().__init__()
        # Local temporal modeling (Conv1D)
        self.conv = nn.Conv1d(in_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm = nn.LayerNorm(hidden_dim)
        # Boundary probability prediction head
        self.fc_boundary = nn.Linear(hidden_dim, 1)
        # Difference feature projection
        self.diff_proj = nn.Linear(in_dim, hidden_dim)
        # Output boundary feature (to be concatenated back to backbone)
        self.out_proj = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x):
        """
        Args:
            x: fused audio-visual features (B, N, D), D=128 by default
        Returns:
            boundary_prob: probability of each frame being a boundary (B, N, 1)
            boundary_feat: boundary features (B, N, out_dim) for backbone concatenation
        """
        B, N, D = x.shape

        # Conv1D (need B, D, N)
        conv_feat = self.conv(x.transpose(1, 2))  # (B, hidden, N)
        conv_feat = conv_feat.transpose(1, 2)     # (B, N, hidden)
        conv_feat = self.norm(conv_feat)
        # Neighbor-frame difference (embedding variation)
        diff = x[:, 1:, :] - x[:, :-1, :] # (B, N-1, D)
        diff = F.pad(diff, (0, 0, 1, 0), mode='constant', value=0) # pad back to (B, N, D)
        diff_feat = self.diff_proj(diff) # (B, N, hidden)

        # Fuse conv + diff
        fuse_feat = torch.cat([conv_feat, diff_feat], dim=-1)  # (B, N, 2*hidden)
        boundary_feat = self.out_proj(fuse_feat)               # (B, N, out_dim=128)

        # Boundary prediction
        boundary_logit = self.fc_boundary(conv_feat)  # (B, N, 1)
        boundary_prob = torch.sigmoid(boundary_logit)

        return boundary_prob, boundary_feat



class MultiModalBoundaryNet_att(nn.Module):
    """
    Two-step cross-attention fusion of three modalities + BoundaryDetectionNet
    """
    def __init__(self, dim_each=128, hidden_dim=128, out_dim=128, kernel_size=5, nhead=4):
        super().__init__()
        self.dim_each = dim_each
        self.nhead = nhead
        
        # MultiheadAttention layers defined directly inside class
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=dim_each, num_heads=nhead, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=dim_each, num_heads=nhead, batch_first=True)

        # BoundaryDetectionNet
        self.boundary_net = BoundaryDetectionNet(
            in_dim=dim_each,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            kernel_size=kernel_size
        )

    def forward(self, vis_feat, vad_feat, spk_feat):
        """
        vis_feat, vad_feat, spk_feat: (B, N, 128)
        Returns:
            boundary_prob: (B, N, 1)
            boundary_feat: (B, N, out_dim)
        """
        # Step 1: vis + vad cross-attention
        fused1, _ = self.cross_attn1(vis_feat, vad_feat, vad_feat)

        # Step 2: fused1 + spk cross-attention
        fused2, _ = self.cross_attn2(fused1, spk_feat, spk_feat)

        # Input BoundaryDetectionNet
        boundary_prob, boundary_feat = self.boundary_net(fused2)

        return boundary_prob, boundary_feat
