import json
import math
import os

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "roneneldan/TinyStories-8M"

VALIDATION_PATH = "data/validation"
RESULTS_PATH = "results/baseline.json"

MAX_LENGTH = 128
BATCH_SIZE = 4

DEVICE = "cpu"


# ---------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# TinyStories tokenizer has no padding token by default.
# We reuse the EOS token for padding.
tokenizer.pad_token = tokenizer.eos_token


# ---------------------------------------------------------
# Load validation dataset
# ---------------------------------------------------------

print("Loading validation dataset...")

validation_dataset = load_from_disk(VALIDATION_PATH)

print(f"Validation stories: {len(validation_dataset)}")


# ---------------------------------------------------------
# Tokenize the validation stories
# ---------------------------------------------------------

def tokenize_batch(batch):
    # Convert raw text into token IDs the model understands.
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


print("Tokenizing validation dataset...")

tokenized_validation = validation_dataset.map(
    tokenize_batch,
    batched=True,

    # Remove the raw text because we only need the tokens now.
    remove_columns=validation_dataset.column_names,
)


# PyTorch DataLoader needs tensors instead of Python lists.
tokenized_validation.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"],
)


# ---------------------------------------------------------
# Create validation DataLoader
# ---------------------------------------------------------

validation_loader = DataLoader(
    tokenized_validation,
    batch_size=BATCH_SIZE,

    # Never shuffle evaluation data.
    shuffle=False,
)


# ---------------------------------------------------------
# Load the original teacher
# ---------------------------------------------------------

print("Loading TinyStories-8M teacher...")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.to(DEVICE)

# Disable dropout and other training-only behaviour.
model.eval()


# Count parameters for information.
num_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(f"Teacher parameters: {num_parameters:,}")


# ---------------------------------------------------------
# Evaluate cross-entropy and perplexity
# ---------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader):

    total_loss = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(dataloader):

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)


        # -------------------------------------------------
        # Create next-token labels
        # -------------------------------------------------

        # For a causal LM, the input itself is also the target.
        labels = input_ids.clone()

        # Ignore padded positions when calculating the loss.
        labels[attention_mask == 0] = -100


        # -------------------------------------------------
        # Forward pass
        # -------------------------------------------------

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Hugging Face automatically shifts the tokens internally:
        #
        # input:  "Once upon a"
        # target:      "upon a time"
        #
        batch_loss = outputs.loss


        # -------------------------------------------------
        # Count valid prediction tokens
        # -------------------------------------------------

        # The first token is not predicted from a previous token.
        # We also exclude all padding positions.
        valid_tokens = (
            labels[:, 1:] != -100
        ).sum().item()


        # outputs.loss is the mean loss for this batch.
        # Multiply by token count so we can compute a proper
        # weighted average over the whole validation set.
        total_loss += batch_loss.item() * valid_tokens

        total_tokens += valid_tokens


        # Print occasional progress.
        if (batch_idx + 1) % 10 == 0:
            print(
                f"Processed {batch_idx + 1} / "
                f"{len(dataloader)} batches"
            )


    # Average negative log-likelihood per token.
    cross_entropy = total_loss / total_tokens

    # Perplexity = e^(cross entropy)
    perplexity = math.exp(cross_entropy)

    return cross_entropy, perplexity, total_tokens


# ---------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------

print("\nEvaluating teacher...\n")

cross_entropy, perplexity, total_tokens = evaluate(
    model,
    validation_loader,
)


# ---------------------------------------------------------
# Show results
# ---------------------------------------------------------

print("\n================================")
print("TinyStories-8M baseline")
print("================================")

print(f"Cross-entropy: {cross_entropy:.4f}")
print(f"Perplexity:    {perplexity:.4f}")
print(f"Tokens tested: {total_tokens:,}")


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results = {
    "model": "teacher_8layer",
    "model_name": MODEL_NAME,
    "parameters": num_parameters,
    "validation_stories": len(validation_dataset),
    "validation_tokens": total_tokens,
    "max_length": MAX_LENGTH,
    "cross_entropy": cross_entropy,
    "perplexity": perplexity,
}


# Create /results if it does not exist.
os.makedirs(
    os.path.dirname(RESULTS_PATH),
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


print(f"\nResults saved to {RESULTS_PATH}")