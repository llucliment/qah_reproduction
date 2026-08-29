import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Fake INT4 quantization
# ---------------------------------------------------------

def fake_quantize_int4(weight):
    """
    Simulate symmetric per-tensor INT4 quantization.

    The underlying parameter remains FP32.
    Only the value used in the forward pass is quantized.

    This is fake quantization, not a native INT4 kernel.
    """

    # INT4 signed range is approximately [-8, 7].
    # We use 7 as the positive scaling limit.
    qmax = 7

    # Find the largest absolute weight value.
    max_abs = weight.detach().abs().amax()

    # Avoid division by zero for an all-zero tensor.
    scale = torch.clamp(
        max_abs / qmax,
        min=1e-8,
    )

    # Map FP32 weights onto integer-like INT4 levels.
    quantized = torch.round(
        weight / scale
    )

    # Clamp to the signed 4-bit range.
    quantized = torch.clamp(
        quantized,
        -8,
        7,
    )

    # Dequantize back to floating point for normal CPU matmul.
    dequantized = quantized * scale

    # Straight-Through Estimator (STE):
    #
    # Forward:
    #     uses dequantized weights
    #
    # Backward:
    #     gradient behaves approximately as if quantization
    #     were the identity function.
    return weight + (
        dequantized - weight
    ).detach()


# ---------------------------------------------------------
# Fake-quantized Linear layer
# ---------------------------------------------------------

class FakeQuantLinear(nn.Linear):
    """
    nn.Linear whose weight is fake-quantized to INT4
    during every forward pass.
    """

    def forward(self, input):
        # Quantize only the weight used in this forward pass.
        quantized_weight = fake_quantize_int4(
            self.weight
        )

        # Bias stays in full precision.
        return F.linear(
            input,
            quantized_weight,
            self.bias,
        )


# ---------------------------------------------------------
# Convert an existing Linear layer
# ---------------------------------------------------------

def convert_linear_to_fake_int4(linear):
    """
    Create FakeQuantLinear with the same trained parameters
    as an existing nn.Linear layer.
    """

    fake_linear = FakeQuantLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
    )

    # Reuse the existing trained parameters.
    #
    # We assign Parameter objects directly instead of making
    # random new parameters and then training from scratch.
    fake_linear.weight = linear.weight

    if linear.bias is not None:
        fake_linear.bias = linear.bias

    return fake_linear


# ---------------------------------------------------------
# Replace Linear modules recursively
# ---------------------------------------------------------

def apply_fake_int4(
    model,
    excluded_module_names=None,
):
    """
    Replace selected nn.Linear layers with FakeQuantLinear.

    Returns:
        model
        list of module names that were quantized
    """

    if excluded_module_names is None:
        excluded_module_names = set()

    else:
        excluded_module_names = set(
            excluded_module_names
        )

    quantized_modules = []


    def replace_recursively(
        parent_module,
        prefix="",
    ):
        # list(...) is important because we modify modules
        # while iterating through their children.
        for child_name, child in list(
            parent_module.named_children()
        ):

            full_name = (
                f"{prefix}.{child_name}"
                if prefix
                else child_name
            )

            # Keep explicitly excluded modules in FP32.
            if full_name in excluded_module_names:
                continue

            if isinstance(
                child,
                FakeQuantLinear,
            ):
                # Avoid quantizing a layer twice.
                continue

            if isinstance(
                child,
                nn.Linear,
            ):
                fake_linear = (
                    convert_linear_to_fake_int4(
                        child
                    )
                )

                setattr(
                    parent_module,
                    child_name,
                    fake_linear,
                )

                quantized_modules.append(
                    full_name
                )

            else:
                # Search deeper inside nested modules.
                replace_recursively(
                    child,
                    full_name,
                )


    replace_recursively(
        model
    )

    return (
        model,
        quantized_modules,
    )
