from datasets import load_dataset


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Start small while we are debugging the pipeline.
# We can increase these numbers later.
NUM_TRAIN = 5000
NUM_VALIDATION = 500

TRAIN_OUTPUT = "data/train"
VALIDATION_OUTPUT = "data/validation"


# ---------------------------------------------------------
# Load TinyStories as a streaming dataset
# ---------------------------------------------------------

print("Connecting to TinyStories dataset...")

# streaming=True means we do NOT download the full dataset.
# Examples are downloaded only as we iterate through them.
train_stream = load_dataset(
    "roneneldan/TinyStories",
    split="train",
    streaming=True,
)

validation_stream = load_dataset(
    "roneneldan/TinyStories",
    split="validation",
    streaming=True,
)

print("Dataset streams loaded.")


# ---------------------------------------------------------
# Shuffle the streams
# ---------------------------------------------------------

# We use a fixed seed so that every run is reproducible.
# The buffer controls how many examples are temporarily held
# when approximating a shuffle of a streaming dataset.
train_stream = train_stream.shuffle(
    seed=42,
    buffer_size=10_000,
)

validation_stream = validation_stream.shuffle(
    seed=42,
    buffer_size=2_000,
)


# ---------------------------------------------------------
# Take only the number of examples we need
# ---------------------------------------------------------

print(f"Downloading {NUM_TRAIN} training stories...")

train_examples = list(
    train_stream.take(NUM_TRAIN)
)

print(f"Downloading {NUM_VALIDATION} validation stories...")

validation_examples = list(
    validation_stream.take(NUM_VALIDATION)
)


# ---------------------------------------------------------
# Convert lists into normal Hugging Face datasets
# ---------------------------------------------------------

# Streaming datasets cannot directly be saved in the same way
# as normal datasets, so we convert them first.
from datasets import Dataset

train_dataset = Dataset.from_list(train_examples)
validation_dataset = Dataset.from_list(validation_examples)


# ---------------------------------------------------------
# Inspect what we downloaded
# ---------------------------------------------------------

print("\nDataset information:")
print(f"Training examples:   {len(train_dataset)}")
print(f"Validation examples: {len(validation_dataset)}")

print("\nColumns:")
print(train_dataset.column_names)

print("\nFirst training story:")
print("----------------------------------------")
print(train_dataset[0]["text"][:1000])
print("----------------------------------------")


# ---------------------------------------------------------
# Save datasets locally
# ---------------------------------------------------------

# save_to_disk automatically creates these directories:
#
# data/train/
# data/validation/

print("\nSaving training dataset...")

train_dataset.save_to_disk(
    TRAIN_OUTPUT
)

print("Saving validation dataset...")

validation_dataset.save_to_disk(
    VALIDATION_OUTPUT
)


# ---------------------------------------------------------
# Finished
# ---------------------------------------------------------

print("\nDone!")

print(f"Training data saved to:   {TRAIN_OUTPUT}")
print(f"Validation data saved to: {VALIDATION_OUTPUT}")