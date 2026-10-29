import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, eps=1,):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits, target, return_components=False):
        # Match target spatial dimensions to the model output.
        # This U-Net uses valid convolutions, so its output can be smaller.
        out_h, out_w = logits.shape[-2:]
        target_h, target_w = target.shape[-2:]

        if out_h > target_h or out_w > target_w:
            raise ValueError(
                f"Output size {out_h}x{out_w} exceeds target "
                f"size {target_h}x{target_w}."
            )

        top = (target_h - out_h) // 2
        left = (target_w - out_w) // 2
        target = target[..., top:top + out_h, left:left + out_w]

        # Convert binary masks to background/foreground channels.
        target = target.float()

        # BCE uses raw logits.
        bce = F.binary_cross_entropy_with_logits(logits, target)

        # Dice uses sigmoid probabilities for both channels.
        probabilities = torch.sigmoid(logits)

        dims = (1, 2, 3)
        intersection = (probabilities * target).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + target.sum(dim=dims)

        dice = (2.0 * intersection + self.eps) / (
            denominator + self.eps
        )
        dice_loss = 1.0 - dice.mean()

        bce_contribution = self.bce_weight * bce
        dice_contribution = self.dice_weight * dice_loss
        total_loss = bce_contribution + dice_contribution

        if return_components:
            components = {
                "bce_loss": bce.detach(),
                "dice_loss": dice_loss.detach(),
                "bce_contribution": bce_contribution.detach(),
                "dice_contribution": dice_contribution.detach(),
            }
            return total_loss, components

        return total_loss