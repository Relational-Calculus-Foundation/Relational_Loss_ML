import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalMSELoss(nn.Module):
    """
    Relational Mean Squared Error Loss.
    
    Transforms standard absolute targets into dimensionless relational ratios 
    anchored to the system's intrinsic capacity (North Star) before calculating the MSE.
    This eliminates environmental scale entropy, preventing exploding gradients 
    and perfectly conditioning the Hessian landscape.
    
    Args:
        eps (float): A small value added to the denominator for numerical stability. Default: 1e-8.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    
    def __init__(self, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_ratio: torch.Tensor, target_absolute: torch.Tensor, capacity: torch.Tensor) -> torch.Tensor:
        """
        Computes the Relational MSE.
        
        Args:
            pred_ratio (torch.Tensor): The model's prediction, structurally bound to [0, 1] 
                                       (e.g., via Sigmoid or ReLU).
            target_absolute (torch.Tensor): The raw target values in human/environmental units.
            capacity (torch.Tensor): The intrinsic theoretical maximum (North Star) for each sample.
            
        Returns:
            torch.Tensor: The dimensionless scalar loss.
            
        Raises:
            RuntimeError: If the shapes of the three tensors do not perfectly match.
        """
        # 1. Structural Validation (Prevents silent broadcasting bugs)
        if pred_ratio.shape != target_absolute.shape or pred_ratio.shape != capacity.shape:
            raise RuntimeError(
                f"Shape mismatch in RelationalMSELoss: "
                f"pred_ratio {pred_ratio.shape}, target_absolute {target_absolute.shape}, "
                f"capacity {capacity.shape} must be strictly identical."
            )
            
        # 2. Relational Translation (Absolute -> Dimensionless)
        target_ratio = target_absolute / (capacity + self.eps)
        
        # 3. Geometric Loss Computation
        return F.mse_loss(pred_ratio, target_ratio, reduction=self.reduction)


class RelationalCrossEntropyLoss(nn.Module):
    """
    Relational Cross Entropy (Binary Cross Entropy with Logits).
    
    Used when the network outputs raw logits rather than probabilities, but the target 
    is a continuous physical or relational ratio bounded in [0, 1].
    
    Args:
        eps (float): A small value added to the denominator for numerical stability. Default: 1e-8.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    
    def __init__(self, eps: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_logits: torch.Tensor, target_absolute: torch.Tensor, capacity: torch.Tensor) -> torch.Tensor:
        """
        Computes the Relational BCE with Logits.
        """
        if pred_logits.shape != target_absolute.shape or pred_logits.shape != capacity.shape:
            raise RuntimeError(
                f"Shape mismatch in RelationalCrossEntropyLoss: "
                f"pred_logits {pred_logits.shape}, target_absolute {target_absolute.shape}, "
                f"capacity {capacity.shape} must be strictly identical."
            )
            
        target_ratio = target_absolute / (capacity + self.eps)
        
        # We use BCEWithLogits because relational targets are conceptually probabilities/ratios [0,1]
        return F.binary_cross_entropy_with_logits(pred_logits, target_ratio, reduction=self.reduction)
