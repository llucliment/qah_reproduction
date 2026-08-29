import copy
from pathlib import Path

import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "roneneldan/TinyStories-8M"
OUTPUT_DIR = "checkpoints/pruned"

# Keep 4 of the original 8 transformer blocks.
# These are the even-numbered layers: 0, 2, 4, 6.
LAYERS_TO_KEEP = [0, 2, 4, 6]


# ---------------------------------------------------------
# Load the original teacher
# ---------------------------------------------------------

print(f"Loading teacher: {MODEL_NAME}")

teacher = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

teacher.eval()

original_num_layers = len(teacher.transformer.h)

print(f"Original transformer layers: {original_num_layers}")
print(f"Keeping layers: {LAYERS_TO_KEEP}")


# ---------------------------------------------------------
# Validate the requested layer indices
# ---------------------------------------------------------

# This avoids silently selecting an invalid layer.
if max(LAYERS_TO_KEEP) >= original_num_layers:
    raise ValueError(
        f"Cannot keep layer {max(LAYERS_TO_KEEP)} because the "
        f"teacher only has {original_num_layers} layers."
    )

# We do not want duplicate layer indices.
if len(set(LAYERS_TO_KEEP)) != len(LAYERS_TO_KEEP):
    raise ValueError("LAYERS_TO_KEEP contains duplicate indices.")


# ---------------------------------------------------------
# Create the structurally compressed student
# ---------------------------------------------------------

# Start from a full copy so embeddings, LM head, layer norms, etc.
# are inherited directly from the teacher.
student = copy.deepcopy(teacher)

# Replace the 8-block transformer stack with copies of only
# layers 0, 2, 4 and 6 from the original teacher.
student.transformer.h = nn.ModuleList(
    [
        copy.deepcopy(teacher.transformer.h[layer_idx])
        for layer_idx in LAYERS_TO_KEEP
    ]
)

num_student_layers = len(LAYERS_TO_KEEP)


# ---------------------------------------------------------
# Update the GPT-Neo configuration
# ---------------------------------------------------------

# GPT-Neo stores both the number of layers and the attention
# pattern in its config. Both must match the pruned architecture
# so that save_pretrained() can later be reloaded correctly.
student.config.num_layers = num_student_layers

# TinyStories-8M alternates global/local attention.
# Layers 0, 2, 4 and 6 are all global-attention blocks,
# so the new 4-layer model contains four global blocks.
student.config.attention_types = [
    [["global"], num_student_layers]
]
student.config.attention_layers = [
    "global"
] * num_student_layers

# The transformer references the model config as well.
# Set these explicitly so the saved checkpoint is self-consistent.
student.transformer.config.num_layers = num_student_layers
student.transformer.config.attention_types = [
    [["global"], num_student_layers]
]
student.transformer.config.attention_layers = [
    "global"
] * num_student_layers


# ---------------------------------------------------------
# Compare parameter counts
# ---------------------------------------------------------

teacher_params = sum(
    parameter.numel()
    for parameter in teacher.parameters()
)

student_params = sum(
    parameter.numel()
    for parameter in student.parameters()
)

print(f"\nTeacher parameters: {teacher_params:,}")
print(f"Student parameters: {student_params:,}")

compression_ratio = student_params / teacher_params

print(
    f"Student / teacher parameter ratio: "
    f"{compression_ratio:.3f}"
)


# ---------------------------------------------------------
# Save the pruned checkpoint
# ---------------------------------------------------------

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

student.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nPruned model saved to: {OUTPUT_DIR}")


# ---------------------------------------------------------
# Verify that the checkpoint reloads correctly
# ---------------------------------------------------------

# This catches config mistakes immediately rather than later
# during recovery/QAT/QAD/QAH.
print("\nReloading checkpoint for verification...")

reloaded = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)

reloaded_num_layers = len(reloaded.transformer.h)

print(f"Reloaded transformer layers: {reloaded_num_layers}")

if reloaded_num_layers != num_student_layers:
    raise RuntimeError(
        "Saved checkpoint does not contain the expected "
        f"{num_student_layers} transformer layers."
    )

print("Pruned checkpoint verified successfully.")
