"""Evidential Deep Learning for uncertainty estimation in mine detection.

Evidential Deep Learning models classification as a Dirichlet distribution
over class probabilities, providing both aleatoric and epistemic uncertainty.

Key concepts:
- Evidence: How much support the model has for each class
- Dirichlet distribution: Distribution over probability distributions
- Epistemic uncertainty: Model uncertainty (reducible with more data)
- Aleatoric uncertainty: Data uncertainty (irreducible noise)

For mine detection, uncertainty estimation is critical:
- High confidence detections → autonomous response
- Low confidence detections → request human verification
- Epistemic uncertainty → model needs more training data
- Aleatoric uncertainty → inherently ambiguous cases

Paper: Evidential Deep Learning to Quantify Classification Uncertainty
       https://arxiv.org/abs/1806.01768
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class EvidentialLayer(nn.Module):
    """Evidential output layer for classification with uncertainty.

    Replaces standard softmax classification with evidential approach
    that outputs Dirichlet parameters (evidence) for uncertainty estimation.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int
    ):
        """Initialize evidential layer.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of classes
        """
        super().__init__()

        self.num_classes = num_classes

        # Linear layer outputs evidence (positive values)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [B, D]

        Returns:
            Evidence (alpha - 1) for each class [B, num_classes]
        """
        # Get evidence (must be non-negative)
        evidence = F.relu(self.linear(x))

        return evidence


def dirichlet_parameters(evidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Dirichlet distribution parameters from evidence.

    Args:
        evidence: Evidence for each class [B, K]

    Returns:
        Tuple of:
            - alpha: Dirichlet parameters [B, K] (alpha = evidence + 1)
            - S: Dirichlet strength [B] (sum of alpha)
    """
    alpha = evidence + 1.0  # Dirichlet parameters
    S = alpha.sum(dim=-1, keepdim=True)  # Total evidence

    return alpha, S


def compute_uncertainty(
    evidence: torch.Tensor,
    num_classes: int
) -> Dict[str, torch.Tensor]:
    """Compute uncertainty metrics from evidence.

    Args:
        evidence: Evidence for each class [B, K]
        num_classes: Number of classes

    Returns:
        Dictionary with:
            - alpha: Dirichlet parameters [B, K]
            - S: Dirichlet strength [B]
            - prob: Expected probability [B, K]
            - uncertainty: Total uncertainty [B]
            - epistemic: Epistemic uncertainty (model) [B]
            - aleatoric: Aleatoric uncertainty (data) [B]
    """
    alpha, S = dirichlet_parameters(evidence)

    # Expected probability (mean of Dirichlet)
    prob = alpha / S

    # Total uncertainty (entropy of expected categorical)
    uncertainty = num_classes / S.squeeze(-1)

    # Epistemic uncertainty (expected entropy of Dirichlet)
    # Higher when model is uncertain about the right class
    epistemic = num_classes / S.squeeze(-1)

    # Aleatoric uncertainty (entropy of expected categorical)
    # Higher when data is inherently ambiguous
    aleatoric = -(prob * torch.log(prob + 1e-10)).sum(dim=-1)

    return {
        "alpha": alpha,
        "S": S,
        "prob": prob,
        "uncertainty": uncertainty,
        "epistemic": epistemic,
        "aleatoric": aleatoric
    }


def kl_divergence(
    alpha: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """Compute KL divergence from Dirichlet to uniform prior.

    This regularizes the model to be uncertain when evidence is weak.

    Args:
        alpha: Dirichlet parameters [B, K]
        num_classes: Number of classes

    Returns:
        KL divergence [B]
    """
    # Uniform Dirichlet prior
    beta = torch.ones_like(alpha)

    S_alpha = alpha.sum(dim=-1, keepdim=True)
    S_beta = beta.sum(dim=-1, keepdim=True)

    # Log normalization constants
    lnB_alpha = torch.lgamma(alpha).sum(dim=-1) - torch.lgamma(S_alpha.squeeze(-1))
    lnB_beta = torch.lgamma(beta).sum(dim=-1) - torch.lgamma(S_beta.squeeze(-1))

    # Digamma terms
    dg_alpha = torch.digamma(alpha + 1e-10)
    dg_S_alpha = torch.digamma(S_alpha + 1e-10)

    # KL divergence
    kl = (
        lnB_alpha - lnB_beta +
        ((alpha - beta) * (dg_alpha - dg_S_alpha)).sum(dim=-1)
    )

    return kl


def evidential_loss(
    evidence: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    lambda_reg: float = 0.01
) -> Dict[str, torch.Tensor]:
    """Compute evidential loss for classification.

    Loss = MSE loss + KL regularization

    MSE loss: Encourages correct predictions
    KL regularization: Encourages uncertainty when wrong

    Args:
        evidence: Evidence for each class [B, K]
        targets: Ground truth class indices [B]
        num_classes: Number of classes
        lambda_reg: KL regularization weight

    Returns:
        Dictionary with:
            - total_loss: Total loss
            - mse_loss: MSE component
            - kl_loss: KL divergence component
    """
    alpha, S = dirichlet_parameters(evidence)

    # One-hot encode targets
    targets_one_hot = F.one_hot(targets, num_classes).float()

    # Expected probability
    prob = alpha / S

    # MSE loss (encourages correct predictions)
    mse_loss = torch.sum((targets_one_hot - prob) ** 2, dim=-1)

    # Variance of predicted probability (uncertainty)
    variance = prob * (1 - prob) / (S + 1)
    mse_loss = mse_loss + torch.sum(variance, dim=-1)

    # KL divergence regularization (encourages uncertainty when wrong)
    kl_loss = kl_divergence(alpha, num_classes)

    # Only apply KL to misclassified samples (annealing)
    pred_class = prob.argmax(dim=-1)
    kl_mask = (pred_class != targets).float()
    kl_loss = kl_loss * kl_mask

    # Total loss
    total_loss = mse_loss + lambda_reg * kl_loss

    return {
        "total_loss": total_loss.mean(),
        "mse_loss": mse_loss.mean(),
        "kl_loss": kl_loss.mean()
    }


class EvidentialClassifier(nn.Module):
    """Complete classifier with evidential output.

    Wraps a feature extractor with evidential layer for
    uncertainty-aware classification.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_dim: int,
        num_classes: int
    ):
        """Initialize evidential classifier.

        Args:
            feature_extractor: Feature extraction network
            feature_dim: Feature dimension
            num_classes: Number of classes
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.evidential_layer = EvidentialLayer(feature_dim, num_classes)
        self.num_classes = num_classes

    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input [B, ...]
            return_uncertainty: Return uncertainty metrics

        Returns:
            Dictionary with evidence and optionally uncertainty metrics
        """
        # Extract features
        features = self.feature_extractor(x)

        # Get evidence
        evidence = self.evidential_layer(features)

        output = {"evidence": evidence}

        # Compute uncertainty if requested
        if return_uncertainty:
            uncertainty_metrics = compute_uncertainty(evidence, self.num_classes)
            output.update(uncertainty_metrics)

        return output

    def predict(
        self,
        x: torch.Tensor,
        uncertainty_threshold: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """Predict with uncertainty filtering.

        Args:
            x: Input [B, ...]
            uncertainty_threshold: Reject predictions above this uncertainty

        Returns:
            Dictionary with predictions and uncertainty
        """
        self.eval()

        with torch.no_grad():
            outputs = self.forward(x, return_uncertainty=True)

        prob = outputs["prob"]
        uncertainty = outputs["uncertainty"]

        # Get predictions
        pred_class = prob.argmax(dim=-1)
        pred_conf = prob.max(dim=-1)[0]

        # Apply uncertainty threshold if provided
        if uncertainty_threshold is not None:
            low_conf_mask = uncertainty > uncertainty_threshold
            pred_class[low_conf_mask] = -1  # Mark as uncertain

        return {
            "predictions": pred_class,
            "confidence": pred_conf,
            "uncertainty": uncertainty,
            "epistemic": outputs["epistemic"],
            "aleatoric": outputs["aleatoric"]
        }


def main():
    """Test evidential deep learning."""
    print("Testing evidential deep learning...")

    # Test evidential layer
    batch_size = 8
    num_classes = 8
    feature_dim = 256

    layer = EvidentialLayer(feature_dim, num_classes)

    # Forward pass
    features = torch.randn(batch_size, feature_dim)
    evidence = layer(features)

    print(f"\nEvidence shape: {evidence.shape}")
    print(f"Evidence range: [{evidence.min():.3f}, {evidence.max():.3f}]")

    # Compute uncertainty
    uncertainty_metrics = compute_uncertainty(evidence, num_classes)

    print(f"\nUncertainty metrics:")
    print(f"  Dirichlet strength: {uncertainty_metrics['S'].squeeze().mean():.3f}")
    print(f"  Mean probability: {uncertainty_metrics['prob'].mean():.3f}")
    print(f"  Total uncertainty: {uncertainty_metrics['uncertainty'].mean():.3f}")
    print(f"  Epistemic: {uncertainty_metrics['epistemic'].mean():.3f}")
    print(f"  Aleatoric: {uncertainty_metrics['aleatoric'].mean():.3f}")

    # Test loss
    targets = torch.randint(0, num_classes, (batch_size,))
    loss_dict = evidential_loss(evidence, targets, num_classes)

    print(f"\nLoss components:")
    print(f"  Total loss: {loss_dict['total_loss']:.4f}")
    print(f"  MSE loss: {loss_dict['mse_loss']:.4f}")
    print(f"  KL loss: {loss_dict['kl_loss']:.4f}")

    # Test with high confidence (correct prediction)
    print("\nHigh confidence scenario:")
    high_conf_evidence = torch.zeros(1, num_classes)
    high_conf_evidence[0, 0] = 100.0  # Very high evidence for class 0
    high_conf_target = torch.tensor([0])

    high_conf_uncertainty = compute_uncertainty(high_conf_evidence, num_classes)
    high_conf_loss = evidential_loss(high_conf_evidence, high_conf_target, num_classes)

    print(f"  Uncertainty: {high_conf_uncertainty['uncertainty'].item():.4f}")
    print(f"  Probability: {high_conf_uncertainty['prob'][0, 0].item():.4f}")
    print(f"  Loss: {high_conf_loss['total_loss'].item():.4f}")

    # Test with low confidence (uncertain)
    print("\nLow confidence scenario:")
    low_conf_evidence = torch.ones(1, num_classes) * 0.5  # Weak evidence
    low_conf_target = torch.tensor([0])

    low_conf_uncertainty = compute_uncertainty(low_conf_evidence, num_classes)
    low_conf_loss = evidential_loss(low_conf_evidence, low_conf_target, num_classes)

    print(f"  Uncertainty: {low_conf_uncertainty['uncertainty'].item():.4f}")
    print(f"  Probability: {low_conf_uncertainty['prob'][0, 0].item():.4f}")
    print(f"  Loss: {low_conf_loss['total_loss'].item():.4f}")

    print("\nEvidential deep learning test successful!")


if __name__ == "__main__":
    main()
