import csv
import math
import os
import random

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization import apply_fake_int4


# ---------------------------------------------------------
# Configuration
# ------------------------------------------+---------------

MODEL_PATH = "checkpoints/recovered"

TRAIN_DATA_PATH = "data/tokenized_train"
VAL_DATA_PATH = "data/tokenized_validation"

OUTPUT_DIR = "checkpoints/qat"
HISTORY_PATH = "results/qat_history.csv"

DEVICE = "cpu"

BATCH_SIZE = 2
LEARNING_RATE = 1e-5

# Start small to verify that the QAT pipeline works.
NUM_STEPS = 100

# Evaluate fake-INT4 validation performance every N steps.
EVAL_EVERY = 20

# Prevent unusually large updates.
MAX_GRAD_NORM = 1.0

SEED = 42


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------
# Load recovered model
# ---------------------------------------------------------

print(
    f"Loading recovered model from: "
    f"{MODEL_PATH}"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)


# ---------------------------------------------------------
# Apply fake INT4 quantization
# ---------------------------------------------------------

# Keep the LM head in FP32 for the same setup used in PTQ.
#
# Embeddings and LayerNorms are not nn.Linear modules,
# so apply_fake_int4() leaves them unchanged automatically.
EXCLUDED_MODULES = {
    "lm_head",
}

model, quantized_modules = apply_fake_int4(
    model,
    excluded_module_names=EXCLUDED_MODULES,
)

model.train()

print(
    f"Fake-INT4 Linear layers: "
    f"{len(quantized_modules)}"
)

for module_name in quantized_modules:
    print(
        f"  quantized: "
        f"{module_name}"
    )


# ---------------------------------------------------------
# Load already-tokenized datasets
# ---------------------------------------------------------

print("\nLoading tokenized datasets...")

train_dataset = load_from_disk(
    TRAIN_DATA_PATH
)

val_dataset = load_from_disk(
    VAL_DATA_PATH
)


# Return stored columns as PyTorch tensors.
train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
    ],
)

val_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
    ],
)


print(
    f"Training examples: "
    f"{len(train_dataset)}"
)

print(
    f"Validation examples: "
    f"{len(val_dataset)}"
)


# ---------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------

# Fixed generator makes shuffle order reproducible.
generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,

    # Never shuffle validation data.
    shuffle=False,
)


# ---------------------------------------------------------
# Validation function
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model,
    dataloader,
):
    """
    Evaluate the model while fake INT4 is active.

    We use real next-token labels, so this gives an
    independent CE/PPL measure of language-model quality.
    """

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        # Causal-LM labels are the same token sequence.
        labels = input_ids.clone()

        # Ignore padding in the loss.
        labels[
            attention_mask == 0
        ] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Hugging Face performs the next-token shift internally.
        valid_tokens = (
            labels[:, 1:] != -100
        ).sum().item()

        total_loss += (
            outputs.loss.item()
            * valid_tokens
        )

        total_tokens += valid_tokens

    cross_entropy = (
        total_loss
        / total_tokens
    )

    perplexity = math.exp(
        cross_entropy
    )

    # Continue training after validation.
    model.train()

    return (
        cross_entropy,
        perplexity,
    )


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

# The stored parameters are still FP32.
#
# FakeQuantLinear only changes the weights used in forward
# passes; the optimizer updates the underlying FP32 weights.
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0,
)


# ---------------------------------------------------------
# Prepare output directories
# ---------------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

os.makedirs(
    os.path.dirname(HISTORY_PATH),
    exist_ok=True,
)


# ---------------------------------------------------------
# Evaluate BEFORE QAT
# ---------------------------------------------------------

print(
    "\nEvaluating fake-INT4 model before QAT..."
)

initial_ce, initial_ppl = evaluate_perplexity(
    model,
    val_loader,
)

print(
    f"Initial validation CE:  "
    f"{initial_ce:.4f}"
)

print(
    f"Initial validation PPL: "
    f"{initial_ppl:.4f}"
)


# ---------------------------------------------------------
# Training history
# ---------------------------------------------------------

history = [
    {
        "step": 0,
        "train_ce": "",
        "validation_ce": initial_ce,
        "validation_ppl": initial_ppl,
    }
]


# ---------------------------------------------------------
# QAT training loop
# ---------------------------------------------------------

print("\nStarting QAT...\n")

step = 0

# Cycle through the dataset until NUM_STEPS is reached.
while step < NUM_STEPS:

    for batch in train_loader:

        if step >= NUM_STEPS:
            break

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)


        # -------------------------------------------------
        # Create hard next-token labels
        # -------------------------------------------------

        labels = input_ids.clone()

        # Padding should not contribute to CE.
        labels[
            attention_mask == 0
        ] = -100


        # -------------------------------------------------
        # Fake-INT4 forward pass
        # -------------------------------------------------

        # Every FakeQuantLinear quantizes its weight during
        # this forward pass before doing the matrix multiply.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # This is standard hard-label next-token CE.
        loss = outputs.loss


        # -------------------------------------------------
        # Backpropagation
        # -------------------------------------------------

        # Remove gradients left from the previous batch.
        optimizer.zero_grad(
            set_to_none=True
        )

        # STE inside FakeQuantLinear lets gradients reach
        # the underlying FP32 weight parameters.
        loss.backward()

        # Clip unusually large gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        # Update the underlying FP32 parameters.
        optimizer.step()

        step += 1


        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        print(
            f"Step "
            f"{step:>3}/{NUM_STEPS} | "
            f"train CE: "
            f"{loss.item():.6f}"
        )


        # -------------------------------------------------
        # Periodic validation
        # -------------------------------------------------

        if (
            step % EVAL_EVERY == 0
            or step == NUM_STEPS
        ):

            val_ce, val_ppl = evaluate_perplexity(
                model,
                val_loader,
            )

            print(
                f"          validation CE:  "
                f"{val_ce:.4f}"
            )

            print(
                f"          validation PPL: "
                f"{val_ppl:.4f}"
            )

            history.append(
                {
                    "step": step,
                    "train_ce": loss.item(),
                    "validation_ce": val_ce,
                    "validation_ppl": val_ppl,
                }
            )


# ---------------------------------------------------------
# Save the learned QAT weights
# ---------------------------------------------------------

print(
    f"\nSaving QAT checkpoint to: "
    f"{OUTPUT_DIR}"
)

# save_pretrained stores the learned underlying FP32 weights.
#
# When evaluating this checkpoint later, fake INT4 must be
# applied again so inference matches QAT training.
model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ---------------------------------------------------------
# Save training history
# ---------------------------------------------------------

with open(
    HISTORY_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "step",
            "train_ce",
            "validation_ce",
            "validation_ppl",
        ],
    )

    writer.writeheader()
    writer.writerows(history)


print(
    f"QAT history saved to: "
    f"{HISTORY_PATH}"
)


# ---------------------------------------------------------
# Final message
# ---------------------------------------------------------

print("\nQAT complete.")

print(
    "Important: checkpoints/qat stores FP32 latent weights. "
    "Apply fake INT4 again when evaluating it."
)
