import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalMSELoss(nn.Module):
    """
    MSE loss mapped into dimensionless relational space.
    Ideal for Physics, Finance, and Continuous Control (RL).
    """
    def __init__(self, capacity_fn=None, reduction='mean'):
        super().__init__()
        self.capacity_fn = capacity_fn
        self.reduction = reduction

    def forward(self, pred_ratio, target_abs, *args):
        if self.capacity_fn is not None:
            capacity = self.capacity_fn(*args)
        else:
            # If no capacity provided, assumes targets are already relational [0,1]
            capacity = torch.ones_like(target_abs)
            
        target_ratio = target_abs / capacity
        return F.mse_loss(pred_ratio, target_ratio, reduction=self.reduction)


class RelationalCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss anchored to the system's Maximum Entropy.
    Ideal for LLMs, NLP, and Classification tasks.
    
    By normalizing by log(vocab_size), the loss becomes dimensionless [0, 1].
    It measures the relative distance from pure chaos, making it invariant
    to vocabulary size changes and preventing gradient explosion in large LLMs.
    """
    def __init__(self, vocab_size, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # The 'North Star' for a probability distribution is Maximum Entropy
        self.max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32))

    def forward(self, logits, targets):
        # Calculate standard absolute cross-entropy
        absolute_ce_loss = F.cross_entropy(
            logits, 
            targets, 
            ignore_index=self.ignore_index, 
            reduction=self.reduction
        )
        
        # Anchor to the intrinsic capacity (Max Entropy) to make it relational
        relational_ce_loss = absolute_ce_loss / self.max_entropy.to(logits.device)
        return relational_ce_loss


class BoundedRatioModel(nn.Module):
    """
    Helper wrapper to enforce the geometric constraint [0,1] on outputs
    for Relational Regression models.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return torch.sigmoid(self.base_model(x))
