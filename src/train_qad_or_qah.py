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

from quantization import apply_fake_int4


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# I reused qad code for qah, so the names in the logs will all say qad, even if doing qah

TRAIN_DATA_PATH = "data/tokenized_train"
VAL_DATA_PATH = "data/tokenized_validation"
STUDENT_MODEL_PATH = "checkpoints/recovered_best"

# QAD
'''
TEACHER_MODEL_PATH = "checkpoints/recovered_best" # Not quantized version
OUTPUT_DIR = "checkpoints/qad"
HISTORY_PATH = "results/qad_history.csv"
'''

# QAH
TEACHER_MODEL_PATH = "roneneldan/TinyStories-8M"
OUTPUT_DIR = "checkpoints/qah"
HISTORY_PATH = "results/qah_history.csv"

DEVICE = "cpu"

BATCH_SIZE = 2
LEARNING_RATE = 1e-5

# Keep the same training budget as QAT for a fair comparison.
NUM_STEPS = 200

# Evaluate every N optimizer steps.
EVAL_EVERY = 20

# Prevent unusually large gradient updates.
MAX_GRAD_NORM = 1.0

# Distillation temperature.
TEMPERATURE = 1.0

SEED = 42


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    TEACHER_MODEL_PATH
)


# ---------------------------------------------------------
# Load full-precision recovered teacher
# ---------------------------------------------------------

print(
    f"Loading recovered FP32 teacher from: "
    f"{TEACHER_MODEL_PATH}"
)

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_PATH
)

teacher.to(DEVICE)

# Teacher is only used to produce target distributions.
teacher.eval()

# Teacher weights must never be updated.
for parameter in teacher.parameters():
    parameter.requires_grad = False


# ---------------------------------------------------------
# Load fresh student from SAME recovered checkpoint
# ---------------------------------------------------------

print(
    f"Loading fresh student from: "
    f"{STUDENT_MODEL_PATH}"
)

student = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_PATH
)

student.to(DEVICE)


# ---------------------------------------------------------
# Apply fake INT4 ONLY to the student
# ---------------------------------------------------------

# Keep the LM head in FP32, exactly like PTQ and QAT.
#
# Embeddings and LayerNorms are not nn.Linear modules,
# so they remain full precision automatically.
EXCLUDED_MODULES = {
    "lm_head",
}

student, quantized_modules = apply_fake_int4(
    student,
    excluded_module_names=EXCLUDED_MODULES,
)

student.train()


print(
    f"Teacher layers: "
    f"{len(teacher.transformer.h)}"
)

print(
    f"Student layers: "
    f"{len(student.transformer.h)}"
)

print(
    f"Fake-INT4 Linear layers in student: "
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

# Fixed generator makes the shuffle reproducible.
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

    # Validation order stays fixed.
    shuffle=False,
)


# ---------------------------------------------------------
# QAD KL loss
# ---------------------------------------------------------

def masked_next_token_kl(
    teacher_logits,
    student_logits,
    attention_mask,
    temperature=1.0,
):
    """
    Compute KL(teacher || student) for causal next-token
    predictions while ignoring padding positions.

    Teacher:
        recovered 4-layer FP32 model

    Student:
        same recovered 4-layer model + fake INT4
    """

    # Logits at position t predict token t+1.
    #
    # We therefore remove the final position because it has
    # no next token inside the current sequence.
    teacher_logits = teacher_logits[
        :, :-1, :
    ]

    student_logits = student_logits[
        :, :-1, :
    ]


    # A prediction at position t is valid only if token t+1
    # is a real token rather than padding.
    target_mask = attention_mask[
        :, 1:
    ].bool()


    # Apply temperature before softmax.
    teacher_logits = (
        teacher_logits
        / temperature
    )

    student_logits = (
        student_logits
        / temperature
    )


    # Teacher gives the target probability distribution.
    teacher_probs = F.softmax(
        teacher_logits,
        dim=-1,
    )


    # torch.kl_div expects log probabilities as its input.
    student_log_probs = F.log_softmax(
        student_logits,
        dim=-1,
    )


    # Compute KL contribution for every vocabulary token.
    #
    # Shape:
    # [batch, sequence_length - 1, vocabulary_size]
    kl_per_vocab = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    )


    # Sum across vocabulary to get one KL value
    # for each next-token prediction position.
    #
    # Shape:
    # [batch, sequence_length - 1]
    kl_per_token = kl_per_vocab.sum(
        dim=-1
    )


    # Remove padding positions.
    valid_kl = kl_per_token[
        target_mask
    ]


    # Mean KL across all valid next-token positions.
    loss = valid_kl.mean()


    # Standard temperature correction used in distillation.
    #
    # At temperature = 1 this has no numerical effect.
    loss = loss * (
        temperature ** 2
    )

    return loss


# ---------------------------------------------------------
# Validation CE + perplexity
# ---------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model,
    dataloader,
):
    """
    Evaluate the fake-INT4 student against REAL TinyStories
    next tokens.

    This gives an independent metric instead of evaluating
    only the KL objective that QAD directly optimizes.
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


        # For causal LM evaluation, input tokens also serve
        # as the target token sequence.
        labels = input_ids.clone()


        # Hugging Face ignores target labels equal to -100.
        # Therefore padding contributes nothing to CE.
        labels[
            attention_mask == 0
        ] = -100


        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )


        # Position 0 is not itself predicted.
        valid_tokens = (
            labels[:, 1:] != -100
        ).sum().item()


        # outputs.loss is a mean over valid tokens.
        # Weight it before combining batches.
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


    # Return student to training mode.
    model.train()


    return (
        cross_entropy,
        perplexity,
    )


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

# Only the student parameters are passed to the optimizer.
#
# The underlying stored student weights remain FP32, while
# FakeQuantLinear exposes quantized versions in forward passes.
optimizer = torch.optim.AdamW(
    student.parameters(),
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
# Evaluate BEFORE QAD training
# ---------------------------------------------------------

print(
    "\nEvaluating fake-INT4 student before QAD or QAH..."
)

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
# Track best checkpoint
# ---------------------------------------------------------

# Lowest validation perplexity seen so far.
best_val_ppl = initial_ppl

# Step where that best perplexity occurred.
best_step = 0


# ---------------------------------------------------------
# QAH training loop
# ---------------------------------------------------------

print("\nStarting QAD or QAH...\n")

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
        # Full-precision teacher forward pass
        # -------------------------------------------------

        # Teacher is frozen, so no backward graph is needed.
        with torch.no_grad():

            teacher_outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            teacher_logits = (
                teacher_outputs.logits
            )


        # -------------------------------------------------
        # Fake-INT4 student forward pass
        # -------------------------------------------------

        student_outputs = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        student_logits = (
            student_outputs.logits
        )


        # -------------------------------------------------
        # QAD loss
        # -------------------------------------------------

        # Unlike QAT, we do NOT use hard next-token labels.
        #
        # The student learns to reproduce the recovered
        # FP32 teacher's entire probability distribution.
        loss = masked_next_token_kl(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            attention_mask=attention_mask,
            temperature=TEMPERATURE,
        )


        # -------------------------------------------------
        # Backpropagation
        # -------------------------------------------------

        # Clear gradients from previous training step.
        optimizer.zero_grad(
            set_to_none=True
        )


        # STE in FakeQuantLinear allows the KL gradient
        # to reach the underlying FP32 student parameters.
        loss.backward()


        # Prevent very large gradient updates.
        torch.nn.utils.clip_grad_norm_(
            student.parameters(),
            MAX_GRAD_NORM,
        )


        # Update only the student.
        optimizer.step()

        step += 1


        # -------------------------------------------------
        # Training progress
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

        if (step % EVAL_EVERY == 0 or step == NUM_STEPS):

            val_ce, val_ppl = evaluate_perplexity(
                student,
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


    # -----------------------------------------------------
    # Save best checkpoint
    # -----------------------------------------------------

    # If this is the best validation PPL seen so far,
    # save a separate checkpoint.
    if val_ppl < best_val_ppl:

        best_val_ppl = val_ppl
        best_step = step

        student.save_pretrained(
            "checkpoints/qah_best"
        )

        tokenizer.save_pretrained(
            "checkpoints/qah_best"
        )

        print(
            f"          New best checkpoint! "
            f"Step {best_step}, "
            f"PPL = {best_val_ppl:.4f}"
        )


    # -----------------------------------------------------
    # Save metrics to history
    # -----------------------------------------------------

    history.append(
        {
            "step": step,
            "train_kl": loss.item(),
            "validation_ce": val_ce,
            "validation_ppl": val_ppl,
        }
    )


# ---------------------------------------------------------
# Save trained QAD student
# ---------------------------------------------------------

print(
    f"\nSaving QAD checkpoint to: "
    f"{OUTPUT_DIR}"
)


# This stores the learned underlying FP32 latent weights.
#
# To evaluate the QAD model as INT4 later, call
# apply_fake_int4() again after loading this checkpoint.
student.save_pretrained(
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
            "train_kl",
            "validation_ce",
            "validation_ppl",
        ],
    )

    writer.writeheader()
    writer.writerows(history)


print(
    f"QAD history saved to: "
    f"{HISTORY_PATH}"
)


# ---------------------------------------------------------
# Final message
# ---------------------------------------------------------

print("\nQAD complete.")

print(
    "Important: checkpoints/qad or checkpoints/qah contains the learned "
    "FP32 latent weights. Apply fake INT4 again during "
    "evaluation so the model is measured in its quantized setup."
)

print(
    f"\nBest validation PPL: "
    f"{best_val_ppl:.4f}"
)

print(
    f"Best step: "
    f"{best_step}"
)