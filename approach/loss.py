import torch
import torch.nn as nn
import torch.nn.functional as F

class lossAnVall_con(nn.Module):
    def __init__(self, temp=0.07, w00=0.1):
        """
        Args:
            temp: temperature for scaling cosine similarity
            w00: weight for 0/0 frames
        """
        super().__init__()
        self.temp = temp
        self.w00 = w00
        self.eps = 1e-8

    def forward(self, audio_feats, visual_feats, audio_labels, visual_labels):
        """
        Args:
            audio_feats: (B, N, D) audio embeddings
            visual_feats: (B, N, D) visual embeddings
            audio_labels: (M,) binary labels (already flattened)
            visual_labels: (M,) binary labels (already flattened)

        Returns:
            loss: scalar
        """
        B, N, D = audio_feats.shape
        M = B * N

        # flatten feats
        a = audio_feats.reshape(M, D)
        v = visual_feats.reshape(M, D)

        la = audio_labels   # already (M,)
        lv = visual_labels  # already (M,)

        # L2 normalize
        a = F.normalize(a, dim=-1)
        v = F.normalize(v, dim=-1)

        # similarity matrix
        sim = (a @ v.t()) / self.temp  # (M, M)
        exp_sim = torch.exp(sim)

        # build positive mask: same frame type (1/1 strong, 0/0 weak)
        pos_mask = (la.unsqueeze(1) == lv.unsqueeze(0)) & ((la.unsqueeze(1) == 1) | (la.unsqueeze(1) == 0))
        # weight mask: 1 for AV=1, w00 for 0/0
        weight_mask = torch.where(
            la.unsqueeze(1) == 1,
            torch.ones_like(sim),
            self.w00 * torch.ones_like(sim)
        )
        weight_mask = weight_mask * pos_mask.float()

        # Audio->Visual loss
        denom_a2v = exp_sim.sum(dim=1, keepdim=True)
        num_a2v = (exp_sim * weight_mask).sum(dim=1, keepdim=True)
        frac_a2v = (num_a2v + self.eps) / (denom_a2v + self.eps)
        valid_a2v = (num_a2v.squeeze(-1) > 0).float()
        loss_a2v = - (valid_a2v * torch.log(frac_a2v.squeeze(-1))).sum() / (valid_a2v.sum().clamp(min=1.0))

        # Visual->Audio loss (symmetric)
        denom_v2a = exp_sim.sum(dim=0, keepdim=True)
        num_v2a = (exp_sim * weight_mask).sum(dim=0, keepdim=True)
        frac_v2a = (num_v2a + self.eps) / (denom_v2a + self.eps)
        valid_v2a = (num_v2a.squeeze(0) > 0).float()
        loss_v2a = - (valid_v2a * torch.log(frac_v2a.squeeze(0))).sum() / (valid_v2a.sum().clamp(min=1.0))

        return 0.5 * (loss_a2v + loss_v2a)

class BoundaryLoss(nn.Module):
    def __init__(self, mode='diff', loss_type='focal_dice', 
                 alpha=0.1, gamma=2.0, 
                 lambda_focal=1.0, lambda_dice=0.5):
        super(BoundaryLoss, self).__init__()
        assert mode in ['diff', 'seg'], "mode must be 'diff' or 'seg'"
        assert loss_type in ['bce', 'focal', 'dice', 'focal_dice']
        
        self.mode = mode
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, label):
        B, N, _ = pred.shape

        # --- reshape label ---
        if label.dim() == 1 or (label.dim() == 2 and label.shape[1] == 1 and label.numel() == B * N):
            label = label.view(B, N, 1)
        elif label.dim() == 2:
            label = label.unsqueeze(-1)

        # --- build boundary labels ---
        if self.mode == 'diff':
            boundary_label = torch.abs(label[:, 1:, :] - label[:, :-1, :].float())
            boundary_label = F.pad(boundary_label, (0, 0, 1, 0), value=0)
        else:  # 'seg'
            label_pad = F.pad(label, (0, 0, 1, 1), value=0)
            start = (label_pad[:, 1:-1, :] - label_pad[:, :-2, :]) > 0
            end = (label_pad[:, 1:-1, :] - label_pad[:, 2:, :]) > 0
            boundary_label = (start | end).float()

        # --- BCE ---
        bce = self.bce_loss(pred, boundary_label)

        # --- Focal BCE ---
        if self.loss_type in ['focal', 'focal_dice']:
            pt = torch.clamp(pred, 1e-6, 1 - 1e-6)
            bce_term = - (boundary_label * torch.log(pt) + (1 - boundary_label) * torch.log(1 - pt))
            focal_weight = self.alpha * boundary_label * (1 - pt).pow(self.gamma) + \
                           (1 - self.alpha) * (1 - boundary_label) * pt.pow(self.gamma)
            focal_bce = (focal_weight * bce_term).mean()
        else:
            focal_bce = 0

        # --- Dice ---
        if self.loss_type in ['dice', 'focal_dice']:
            smooth = 1e-6
            pred_flat = pred.view(-1)
            label_flat = boundary_label.view(-1)
            intersection = (pred_flat * label_flat).sum()
            dice_loss = 1 - (2. * intersection + smooth) / (pred_flat.sum() + label_flat.sum() + smooth)
        else:
            dice_loss = 0

        # --- Combine ---
        if self.loss_type == 'bce':
            loss = bce
        elif self.loss_type == 'focal':
            loss = focal_bce
        elif self.loss_type == 'dice':
            loss = dice_loss
        else:  # focal_dice
            loss = self.lambda_focal * focal_bce + self.lambda_dice * dice_loss

        return loss

