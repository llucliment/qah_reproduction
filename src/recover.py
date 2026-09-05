import csv
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TEACHER_MODEL = "roneneldan/TinyStories-8M"
STUDENT_MODEL = "checkpoints/pruned"

TRAIN_DATA_PATH = "data/tokenized_train"
VAL_DATA_PATH = "data/tokenized_validation"

OUTPUT_DIR = "checkpoints/recovered_best"
HISTORY_PATH = "results/recovery_history_best.csv"

DEVICE = "cpu"

BATCH_SIZE = 2
LEARNING_RATE = 1e-5

# Start with 100 steps as a smoke test.
NUM_STEPS = 200

# Evaluate the student every N optimization steps.
EVAL_EVERY = 20

# Gradient clipping helps avoid unstable updates.
MAX_GRAD_NORM = 1.0

# QAH uses temperature = 1.
# We keep the same value for this recovery stage.
TEMPERATURE = 1.0

SEED = 42


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------
# Load teacher and student
# ---------------------------------------------------------

print("Loading original 8-layer teacher...")

# The tokenizer is shared by teacher and student.
tokenizer = AutoTokenizer.from_pretrained(
    TEACHER_MODEL
)

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL
)

teacher.to(DEVICE)

# Teacher is never trained.
teacher.eval()

for parameter in teacher.parameters():
    parameter.requires_grad = False


print("Loading pruned 4-layer student...")

student = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL
)

student.to(DEVICE)

# Student IS trained.
student.train()


print(
    f"Teacher layers: "
    f"{len(teacher.transformer.h)}"
)

print(
    f"Student layers: "
    f"{len(student.transformer.h)}"
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


# Return stored token IDs as PyTorch tensors.
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

# Fixed generator makes the training shuffle reproducible.
generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,

    # Training examples should be shuffled.
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
# KL distillation loss
# ---------------------------------------------------------

def masked_next_token_kl(
    teacher_logits,
    student_logits,
    attention_mask,
    temperature=1.0,
):
    """
    KL(teacher || student) for causal next-token predictions.

    Teacher/student logits have shape:
        [batch, sequence_length, vocabulary]

    We use logits at position t to predict token t+1,
    so the final logit position is removed.

    Padding target positions are masked out.
    """

    # Logits at position t predict the next token t+1.
    teacher_logits = teacher_logits[:, :-1, :]
    student_logits = student_logits[:, :-1, :]

    # A prediction is valid only when its NEXT token is real.
    target_mask = attention_mask[:, 1:].bool()

    # Apply the distillation temperature.
    teacher_logits = teacher_logits / temperature
    student_logits = student_logits / temperature

    # Teacher is the target probability distribution.
    teacher_probs = F.softmax(
        teacher_logits,
        dim=-1,
    )

    # PyTorch kl_div expects log probabilities as its input.
    student_log_probs = F.log_softmax(
        student_logits,
        dim=-1,
    )

    # Compute KL for every vocabulary element.
    kl_per_vocab = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    )

    # Sum across vocabulary to get one KL value per token.
    kl_per_token = kl_per_vocab.sum(
        dim=-1
    )

    # Remove padding positions.
    masked_kl = kl_per_token[
        target_mask
    ]

    # Mean KL over valid next-token positions.
    loss = masked_kl.mean()

    # Standard distillation scaling.
    # At T=1 this changes nothing.
    loss = loss * (
        temperature ** 2
    )

    return loss


# ---------------------------------------------------------
# Validation perplexity
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model,
    dataloader,
):
    """
    Evaluate student on the real TinyStories next tokens.

    This is independent of the teacher and lets us compare
    pruned vs recovered vs later quantized models.
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

        # Causal-LM labels are the input tokens themselves.
        labels = input_ids.clone()

        # Padding must not contribute to cross-entropy.
        labels[
            attention_mask == 0
        ] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Only positions 1...T are actual prediction targets.
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

    # Return the model to training mode afterwards.
    model.train()

    return (
        cross_entropy,
        perplexity,
    )


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

optimizer = torch.optim.AdamW(
    student.parameters(),
    lr=LEARNING_RATE,

    # Keep this simple for the first reproduction.
    weight_decay=0.0,
)


# ---------------------------------------------------------
# Prepare output folders
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
# Measure student BEFORE recovery
# ---------------------------------------------------------

print("\nEvaluating pruned student before recovery...")

initial_ce, initial_ppl = evaluate_perplexity(
    student,
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
        "train_kl": "",
        "validation_ce": initial_ce,
        "validation_ppl": initial_ppl,
    }
]

# ---------------------------------------------------------
# Track best recovery checkpoint
# ---------------------------------------------------------

# Step 0 is already evaluated.
best_val_ppl = initial_ppl
best_step = 0

# ---------------------------------------------------------
# Recovery training loop
# ---------------------------------------------------------

print("\nStarting KL recovery...\n")

step = 0

# Cycle through the DataLoader until NUM_STEPS is reached.
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
        # Teacher forward pass
        # -------------------------------------------------

        # No autograd graph is built for the frozen teacher.
        with torch.no_grad():

            teacher_outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            teacher_logits = (
                teacher_outputs.logits
            )


        # -------------------------------------------------
        # Student forward pass
        # -------------------------------------------------

        student_outputs = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        student_logits = (
            student_outputs.logits
        )


        # -------------------------------------------------
        # Distillation loss
        # -------------------------------------------------

        loss = masked_next_token_kl(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            attention_mask=attention_mask,
            temperature=TEMPERATURE,
        )


        # -------------------------------------------------
        # Backpropagation
        # -------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        # Limit unusually large gradients.
        torch.nn.utils.clip_grad_norm_(
            student.parameters(),
            MAX_GRAD_NORM,
        )

        optimizer.step()

        step += 1


        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        print(
            f"Step "
            f"{step:>3}/{NUM_STEPS} | "
            f"train KL: "
            f"{loss.item():.6f}"
        )


        # -------------------------------------------------
        # Periodic validation
        # -------------------------------------------------

        if (
            step % EVAL_EVERY == 0
            or step == NUM_STEPS
            or step == 1
        ):

            val_ce, val_ppl = evaluate_perplexity(
                student,
                val_loader,
            )

            # Guardar mejor checkpoint SOLO después de evaluar.
            if val_ppl < best_val_ppl:

                best_val_ppl = val_ppl
                best_step = step

                student.save_pretrained(
                    "checkpoints/recovered_best"
                )

                tokenizer.save_pretrained(
                    "checkpoints/recovered_best"
                )

                print(
                    f"          New best checkpoint! "
                    f"Step {best_step}, "
                    f"PPL = {best_val_ppl:.4f}"
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
                    "train_kl": loss.item(),
                    "validation_ce": val_ce,
                    "validation_ppl": val_ppl,
                }
            )


# ---------------------------------------------------------
# Save recovered student
# ---------------------------------------------------------

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
            "train_kl",
            "validation_ce",
            "validation_ppl",
        ],
    )

    writer.writeheader()
    writer.writerows(history)


print(
    f"Recovery history saved to: "
    f"{HISTORY_PATH}"
)


# ---------------------------------------------------------
# Final verification
# ---------------------------------------------------------

print("\nReloading recovered checkpoint...")

reloaded_student = (
    AutoModelForCausalLM.from_pretrained(
        OUTPUT_DIR
    )
)

print(
    f"Recovered layers: "
    f"{len(reloaded_student.transformer.h)}"
)

print("\nRecovery complete.")
