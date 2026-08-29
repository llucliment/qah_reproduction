from datasets import load_from_disk
from transformers import AutoTokenizer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "roneneldan/TinyStories-8M"

MAX_LENGTH = 128

TRAIN_PATH = "data/train"
VAL_PATH = "data/validation"

TOKENIZED_TRAIN_PATH = "data/tokenized_train"
TOKENIZED_VAL_PATH = "data/tokenized_validation"


# ---------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# TinyStories GPT-Neo tokenizer does not define a padding token.
# We can safely use the EOS token as padding.
tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")
print("Vocabulary size:", len(tokenizer))
print("Padding token:", tokenizer.pad_token)
print("EOS token:", tokenizer.eos_token)


# ---------------------------------------------------------
# Load our fixed dataset subsets
# ---------------------------------------------------------

train_dataset = load_from_disk(TRAIN_PATH)
val_dataset = load_from_disk(VAL_PATH)

print(f"\nTraining examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)}")


# ---------------------------------------------------------
# Tokenization function
# ---------------------------------------------------------

def tokenize_function(batch):
    """
    Convert TinyStories text into token IDs.

    Every sequence will have exactly MAX_LENGTH tokens.
    Short sequences are padded and long sequences truncated.
    """

    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


# ---------------------------------------------------------
# Tokenize training dataset
# ---------------------------------------------------------

print("\nTokenizing training data...")

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,

    # We no longer need the original text after tokenization.
    remove_columns=train_dataset.column_names,
)


# ---------------------------------------------------------
# Tokenize validation dataset
# ---------------------------------------------------------

print("Tokenizing validation data...")

tokenized_val = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names,
)


# ---------------------------------------------------------
# Save datasets
# ---------------------------------------------------------

tokenized_train.save_to_disk(TOKENIZED_TRAIN_PATH)
tokenized_val.save_to_disk(TOKENIZED_VAL_PATH)

print("\nTokenized datasets saved.")

print(f"Train:      {TOKENIZED_TRAIN_PATH}")
print(f"Validation: {TOKENIZED_VAL_PATH}")


# ---------------------------------------------------------
# Inspect one example
# ---------------------------------------------------------

example = tokenized_train[0]

print("\nExample:")
print("Number of input tokens:", len(example["input_ids"]))
print("First 20 token IDs:", example["input_ids"][:20])
print("First 20 attention-mask values:", example["attention_mask"][:20])

# Convert tokens back to text to make sure everything looks reasonable.
decoded = tokenizer.decode(
    example["input_ids"],
    skip_special_tokens=True,
)

print("\nDecoded example:")
print(decoded[:500])