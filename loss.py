import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    #returns the Tversky loss per batch
    def __init__(self, smooth = 0.000001, alpha = 0.5, beta = 0.5):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        # Flatten both prediction and GT tensors
        y_pred_flat = torch.flatten(y_pred)
        y_true_flat = torch.flatten(y_true)
        
        tp = (y_pred_flat * y_true_flat).sum()
        fp = (y_pred_flat * (1 - y_true_flat)).sum()
        fn = ((1 - y_pred_flat) * y_true_flat).sum()
        tversky = (tp + self.smooth)/(tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky

