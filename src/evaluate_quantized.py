import json
import math
import os

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from quantization import apply_fake_int4


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VALIDATION_PATH = "data/tokenized_validation"

# Evaluate the weights learned by PTQ
'''
MODEL_PATH = "checkpoints/recovered"
RESULTS_PATH = "results/ptq.json"
'''

# Evaluate the weights learned by QAT.
'''
MODEL_PATH = "checkpoints/qat"
RESULTS_PATH = "results/qat.json"
'''

# Evaluate the weights learned by QAD.
MODEL_PATH = "checkpoints/qad"
RESULTS_PATH = "results/qad.json"

BATCH_SIZE = 4
DEVICE = "cpu"


# ---------------------------------------------------------
# Load recovered FP32 model
# ---------------------------------------------------------

print(
    f"Loading recovered model from: "
    f"{MODEL_PATH}"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()


# ---------------------------------------------------------
# Model information before quantization
# ---------------------------------------------------------

num_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

num_layers = len(
    model.transformer.h
)

print(
    f"Transformer layers: "
    f"{num_layers}"
)

print(
    f"Parameters: "
    f"{num_parameters:,}"
)


# ---------------------------------------------------------
# Apply fake INT4
# ---------------------------------------------------------

# Keep the final LM head in full precision for this first
# controlled experiment.
#
# Embeddings and LayerNorms are not nn.Linear modules, so
# apply_fake_int4() leaves them untouched automatically.
EXCLUDED_MODULES = {
    "lm_head",
}

model, quantized_modules = apply_fake_int4(
    model,
    excluded_module_names=EXCLUDED_MODULES,
)


print(
    f"\nFake-INT4 Linear layers: "
    f"{len(quantized_modules)}"
)

for module_name in quantized_modules:
    print(
        f"  quantized: "
        f"{module_name}"
    )


# ---------------------------------------------------------
# Load PRE-TOKENIZED validation data
# ---------------------------------------------------------

print(
    f"\nLoading validation data from: "
    f"{VALIDATION_PATH}"
)

validation_dataset = load_from_disk(
    VALIDATION_PATH
)

validation_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
    ],
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,

    # Keep validation deterministic.
    shuffle=False,
)


# ---------------------------------------------------------
# Evaluate CE and perplexity
# ---------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    dataloader,
):
    """
    Evaluate next-token cross-entropy and perplexity.

    The model is fake-INT4 during the forward pass.
    """

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(
        dataloader
    ):

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        # Causal language modelling uses input tokens
        # themselves as the next-token targets.
        labels = input_ids.clone()

        # Ignore padded target positions.
        labels[
            attention_mask == 0
        ] = -100


        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        batch_loss = outputs.loss


        # Hugging Face shifts causal-LM labels internally.
        valid_tokens = (
            labels[:, 1:] != -100
        ).sum().item()


        # Weight the mean batch loss by token count.
        total_loss += (
            batch_loss.item()
            * valid_tokens
        )

        total_tokens += (
            valid_tokens
        )


        if (
            batch_idx + 1
        ) % 10 == 0:

            print(
                f"Processed "
                f"{batch_idx + 1} / "
                f"{len(dataloader)} batches"
            )


    cross_entropy = (
        total_loss
        / total_tokens
    )

    perplexity = math.exp(
        cross_entropy
    )

    return (
        cross_entropy,
        perplexity,
        total_tokens,
    )


# ---------------------------------------------------------
# Run PTQ evaluation
# ---------------------------------------------------------

print(
    "\nEvaluating fake-INT4 PTQ model...\n"
)

(
    cross_entropy,
    perplexity,
    total_tokens,
) = evaluate(
    model,
    validation_loader,
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print(
    "\n================================"
)

print(
    "Recovered 4-layer + fake INT4"
)

print(
    "================================"
)

print(
    f"Cross-entropy: "
    f"{cross_entropy:.4f}"
)

print(
    f"Perplexity:    "
    f"{perplexity:.4f}"
)

print(
    f"Tokens tested: "
    f"{total_tokens:,}"
)


# ---------------------------------------------------------
# Save PTQ results
# ---------------------------------------------------------

results = {
    "name": "ptq_fake_int4",
    "base_model": MODEL_PATH,
    "transformer_layers": num_layers,
    "parameters": num_parameters,
    "quantization": (
        "symmetric per-tensor fake INT4"
    ),
    "quantized_linear_layers": (
        len(quantized_modules)
    ),
    "excluded_modules": sorted(
        EXCLUDED_MODULES
    ),
    "validation_examples": len(
        validation_dataset
    ),
    "validation_tokens": total_tokens,
    "cross_entropy": cross_entropy,
    "perplexity": perplexity,
}


output_directory = os.path.dirname(
    RESULTS_PATH
)

if output_directory:
    os.makedirs(
        output_directory,
        exist_ok=True,
    )


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=4,
    )


print(
    f"\nResults saved to: "
    f"{RESULTS_PATH}"
)