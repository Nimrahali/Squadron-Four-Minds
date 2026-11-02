"""
================================================================================
AURA SUBSYSTEM 04: ALIGNMENT PIPELINE CORE
Module: 04_Activation_Steering_and_Audit.py
Station: Security & Ethics Sentinel
================================================================================
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Callable


class ActivationSteeringHook:
    """
    PyTorch forward hook injecting alpha * v_steer into residual stream hidden tensors during inference.
    """
    def __init__(self, target_module: nn.Module, steering_vector: torch.Tensor, alpha: float = 1.0):
        self.target_module = target_module
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.hook_handle = None

    def register(self):
        """Registers the forward hook on target module."""
        self.hook_handle = self.target_module.register_forward_hook(self._hook_fn)

    def remove(self):
        """Removes hook from target module."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def _hook_fn(self, module: nn.Module, input_args: Tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor:
        """
        Modifies residual stream hidden tensor in-place during forward pass.
        """
        # Handle tuple output (hidden_states, key_value_cache, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Add steering vector multiplied by intensity scalar alpha
        steered_hidden = hidden_states + (self.alpha * self.steering_vector.to(hidden_states.device))

        if rest is not None:
            return (steered_hidden,) + rest
        return steered_hidden


class ConceptVectorExtractor:
    """
    Extracts contrast direction vectors via activation differencing over prompt pairs.
    """
    @staticmethod
    def extract_contrast_vector(
        model: nn.Module,
        target_layer: nn.Module,
        positive_tokens: torch.Tensor,
        negative_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes layer activation difference: v_steer = mean(h_pos) - mean(h_neg)
        """
        pos_activations = []
        neg_activations = []

        def pos_hook(module, input_args, output):
            h = output[0] if isinstance(output, tuple) else output
            pos_activations.append(h.detach().cpu())

        def neg_hook(module, input_args, output):
            h = output[0] if isinstance(output, tuple) else output
            neg_activations.append(h.detach().cpu())

        # Extract positive activations
        handle_pos = target_layer.register_forward_hook(pos_hook)
        with torch.no_grad():
            model(positive_tokens)
        handle_pos.remove()

        # Extract negative activations
        handle_neg = target_layer.register_forward_hook(neg_hook)
        with torch.no_grad():
            model(negative_tokens)
        handle_neg.remove()

        mean_pos = pos_activations[0].mean(dim=(0, 1))
        mean_neg = neg_activations[0].mean(dim=(0, 1))

        # Contrast direction vector v_steer
        v_steer = mean_pos - mean_neg
        return v_steer / torch.norm(v_steer)


class AlignmentTaxAuditor:
    """
    Evaluates reasoning capability retention (GSM8K/MMLU) across steering intensities alpha.
    """
    @staticmethod
    def run_alignment_tax_audit(
        eval_fn: Callable[[float], float],
        alpha_range: List[float]
    ) -> Dict[float, float]:
        """
        Runs reasoning benchmark evaluations for each steering intensity alpha.
        """
        results = {}
        for alpha in alpha_range:
            score = eval_fn(alpha)
            results[alpha] = score
            print(f"  ├─ Steering Intensity alpha = {alpha:+4.1f} | Benchmark Score: {score:.2%}")
        return results


if __name__ == "__main__":
    # Simulated Transformer Layer and Residual Stream Tensor
    hidden_dim = 256
    layer_module = nn.Identity()

    torch.manual_seed(42)
    contrast_vector = torch.randn(hidden_dim)
    contrast_vector = contrast_vector / torch.norm(contrast_vector)

    # Register Steering Hook
    steering_hook = ActivationSteeringHook(
        target_module=layer_module,
        steering_vector=contrast_vector,
        alpha=2.5
    )
    steering_hook.register()

    # Test forward pass with hook enabled
    dummy_input = torch.zeros(1, 8, hidden_dim)
    steered_output = layer_module(dummy_input)

    steering_hook.remove()

    print(f"[VERIFICATION] Unsteered Tensor Norm: {dummy_input.norm().item():.4f}")
    print(f"[VERIFICATION] Steered Tensor Norm:   {steered_output.norm().item():.4f}")

    # Dummy Alignment Tax Audit
    def mock_reasoning_eval(alpha: float) -> float:
        # Benchmark score degrades if steering intensity is excessively high
        base_acc = 0.82
        tax_penalty = 0.05 * (alpha ** 2)
        return max(0.0, base_acc - tax_penalty)

    print("\n=== ALIGNMENT TAX BENCHMARK AUDIT ===")
    AlignmentTaxAuditor.run_alignment_tax_audit(mock_reasoning_eval, [-2.0, -1.0, 0.0, 1.0, 2.0])