import argparse
import json
import math
import os

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

# Example:
# python src/evaluate_model.py --model roneneldan/TinyStories-8M --name teacher_8layer --output results/teacher.json
# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VALIDATION_PATH = "data/tokenized_validation"

BATCH_SIZE = 4
DEVICE = "cpu"


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Evaluate a TinyStories causal language model."
)

parser.add_argument(
    "--model",
    required=True,
    help=(
        "Hugging Face model name or local checkpoint path, "
        "for example roneneldan/TinyStories-8M "
        "or checkpoints/pruned."
    ),
)

parser.add_argument(
    "--name",
    required=True,
    help=(
        "Short name used in the results file, "
        "for example teacher_8layer or pruned_4layer."
    ),
)

parser.add_argument(
    "--output",
    required=True,
    help=(
        "Path where the evaluation JSON will be saved, "
        "for example results/pruned.json."
    ),
)

# Read the values supplied in the terminal.
args = parser.parse_args()


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print(f"Loading model from: {args.model}")

model = AutoModelForCausalLM.from_pretrained(
    args.model
)

# We are CPU-only.
model.to(DEVICE)

# Evaluation mode disables dropout and other training behaviour.
model.eval()

print("Model loaded.")


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------

# Count all parameters in the model.
num_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

# GPT-Neo stores its transformer blocks in transformer.h.
num_layers = len(
    model.transformer.h
)

print(f"Transformer layers: {num_layers}")
print(f"Parameters: {num_parameters:,}")


# ---------------------------------------------------------
# Load PRE-TOKENIZED validation dataset
# ---------------------------------------------------------

print(
    f"\nLoading tokenized validation data from: "
    f"{VALIDATION_PATH}"
)

validation_dataset = load_from_disk(
    VALIDATION_PATH
)

print(
    f"Validation examples: "
    f"{len(validation_dataset)}"
)


# ---------------------------------------------------------
# Convert dataset columns to PyTorch tensors
# ---------------------------------------------------------

# tokenize_data.py already created:
#
# input_ids
# attention_mask
#
# We only need to convert those stored lists into tensors.
validation_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
    ],
)


# ---------------------------------------------------------
# Create DataLoader
# ---------------------------------------------------------

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,

    # Never shuffle the validation set.
    shuffle=False,
)


# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    dataloader,
):
    """
    Evaluate a causal language model using:

    1. Cross-entropy
    2. Perplexity

    Lower is better for both metrics.
    """

    # Make sure the model stays in evaluation mode.
    model.eval()

    total_loss = 0.0
    total_tokens = 0


    # -----------------------------------------------------
    # Iterate over validation batches
    # -----------------------------------------------------

    for batch_idx, batch in enumerate(dataloader):

        # Move tensors to the selected device.
        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)


        # -------------------------------------------------
        # Create labels
        # -------------------------------------------------

        # In causal language modelling, the same token
        # sequence is used both as input and as target.
        labels = input_ids.clone()

        # Padding positions should NOT affect the loss.
        #
        # Hugging Face ignores labels with value -100.
        labels[
            attention_mask == 0
        ] = -100


        # -------------------------------------------------
        # Forward pass
        # -------------------------------------------------

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # outputs.loss is next-token cross-entropy.
        #
        # Hugging Face automatically performs the causal
        # one-token shift internally.
        batch_loss = outputs.loss


        # -------------------------------------------------
        # Count valid prediction tokens
        # -------------------------------------------------

        # Example:
        #
        # input:
        # Once | upon | a | time
        #
        # predictions:
        #        upon | a | time
        #
        # Therefore token position 0 is not itself a target.
        #
        # Padding positions are also excluded because they
        # were replaced by -100 above.
        valid_tokens = (
            labels[:, 1:] != -100
        ).sum().item()


        # -------------------------------------------------
        # Accumulate weighted loss
        # -------------------------------------------------

        # outputs.loss is the MEAN loss for the current
        # batch.
        #
        # Multiply by the number of valid tokens so we can
        # correctly average over the full validation set.
        total_loss += (
            batch_loss.item()
            * valid_tokens
        )

        total_tokens += valid_tokens


        # -------------------------------------------------
        # Progress output
        # -------------------------------------------------

        if (
            batch_idx + 1
        ) % 10 == 0:

            print(
                f"Processed "
                f"{batch_idx + 1} / "
                f"{len(dataloader)} batches"
            )


    # -----------------------------------------------------
    # Final metrics
    # -----------------------------------------------------

    # Mean negative log-likelihood per token.
    cross_entropy = (
        total_loss
        / total_tokens
    )

    # Perplexity = e^(cross-entropy)
    perplexity = math.exp(
        cross_entropy
    )

    return (
        cross_entropy,
        perplexity,
        total_tokens,
    )


# ---------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------

print("\nEvaluating model...\n")

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

print("\n================================")
print(args.name)
print("================================")

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
# Prepare results dictionary
# ---------------------------------------------------------

results = {
    "name": args.name,
    "model": args.model,
    "transformer_layers": num_layers,
    "parameters": num_parameters,
    "validation_examples": len(
        validation_dataset
    ),
    "validation_tokens": total_tokens,
    "cross_entropy": cross_entropy,
    "perplexity": perplexity,
}


# ---------------------------------------------------------
# Create output directory if necessary
# ---------------------------------------------------------

output_directory = os.path.dirname(
    args.output
)

# If the path contains a directory such as results/,
# make sure it exists.
if output_directory:
    os.makedirs(
        output_directory,
        exist_ok=True,
    )


# ---------------------------------------------------------
# Save results as JSON
# ---------------------------------------------------------

with open(
    args.output,
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
    f"{args.output}"
)