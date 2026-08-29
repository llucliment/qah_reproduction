import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RESULTS_DIR = Path("results")

# JSON files containing final evaluation metrics.
JSON_FILES = {
    "Recovered FP32": RESULTS_DIR / "recovered.json",
    "PTQ INT4": RESULTS_DIR / "ptq.json",
    "QAT INT4": RESULTS_DIR / "qat.json",
    "QAD INT4": RESULTS_DIR / "qad.json",
    "QAH INT4": RESULTS_DIR / "qah.json",
}

# CSV files containing validation metrics during training.
HISTORY_FILES = {
    "QAT": RESULTS_DIR / "qat_history.csv",
    "QAD": RESULTS_DIR / "qad_history.csv",
    "QAH": RESULTS_DIR / "qah_history.csv",
}

SUMMARY_OUTPUT = RESULTS_DIR / "summary.csv"
RECOVERY_OUTPUT = RESULTS_DIR / "recovery_percentages.csv"
PLOT_OUTPUT = RESULTS_DIR / "ppl_training_curves.png"


# ---------------------------------------------------------
# Helper: load one JSON file
# ---------------------------------------------------------

def load_json(path):
    # Stop with a useful error if a result file is missing.
    if not path.exists():
        raise FileNotFoundError(
            f"Missing result file: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ---------------------------------------------------------
# Load final evaluation results
# ---------------------------------------------------------

results = {}

for model_name, file_path in JSON_FILES.items():
    results[model_name] = load_json(
        file_path
    )


# ---------------------------------------------------------
# Build summary table
# ---------------------------------------------------------

summary_rows = []

for model_name, result in results.items():

    summary_rows.append(
        {
            "model": model_name,
            "cross_entropy": result["cross_entropy"],
            "perplexity": result["perplexity"],
            "parameters": result.get(
                "parameters",
                None,
            ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

# Lower CE/PPL is better.
summary_df = summary_df.sort_values(
    by="perplexity",
    ascending=True,
)


print("\n======================================")
print("Final model comparison")
print("======================================\n")

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# Save the table so it can be used later in reports.
summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ---------------------------------------------------------
# Quantization-damage recovery percentages
# ---------------------------------------------------------

# Recovered FP32 is our pre-quantization reference.
ce_recovered = results[
    "Recovered FP32"
]["cross_entropy"]

# PTQ shows how much CE worsens from quantization alone.
ce_ptq = results[
    "PTQ INT4"
]["cross_entropy"]


quantization_damage = (
    ce_ptq
    - ce_recovered
)


print("\n======================================")
print("Quantization damage")
print("======================================\n")

print(
    f"Recovered FP32 CE: "
    f"{ce_recovered:.4f}"
)

print(
    f"PTQ INT4 CE:       "
    f"{ce_ptq:.4f}"
)

print(
    f"Quantization damage: "
    f"{quantization_damage:.4f}"
)


# Quantization damage should normally be positive.
if quantization_damage <= 0:
    print(
        "\nWarning: PTQ did not increase CE. "
        "Recovery percentages may not be meaningful."
    )


recovery_rows = []

for method in [
    "QAT INT4",
    "QAD INT4",
    "QAH INT4",
]:

    ce_method = results[
        method
    ]["cross_entropy"]

    # Percentage of the PTQ CE degradation that was recovered.
    recovery_fraction = (
        ce_ptq
        - ce_method
    ) / quantization_damage

    recovery_percentage = (
        recovery_fraction
        * 100
    )

    recovery_rows.append(
        {
            "method": method,
            "cross_entropy": ce_method,
            "recovery_percentage": recovery_percentage,
        }
    )


recovery_df = pd.DataFrame(
    recovery_rows
)


print("\n======================================")
print("Quantization damage recovered")
print("======================================\n")

print(
    recovery_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


recovery_df.to_csv(
    RECOVERY_OUTPUT,
    index=False,
)


# ---------------------------------------------------------
# Load training histories
# ---------------------------------------------------------

history_data = {}

for method_name, file_path in HISTORY_FILES.items():

    # Stop early if one experiment has not been run yet.
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing history file: {file_path}"
        )

    history_data[
        method_name
    ] = pd.read_csv(
        file_path
    )


# ---------------------------------------------------------
# Print best validation PPL for each method
# ---------------------------------------------------------

print("\n======================================")
print("Best training checkpoint by PPL")
print("======================================\n")

for method_name, history_df in history_data.items():

    # Find the row with the minimum validation perplexity.
    best_index = history_df[
        "validation_ppl"
    ].idxmin()

    best_row = history_df.loc[
        best_index
    ]

    print(
        f"{method_name}: "
        f"PPL = {best_row['validation_ppl']:.4f} "
        f"at step {int(best_row['step'])}"
    )


# ---------------------------------------------------------
# Plot validation PPL vs training step
# ---------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)


for method_name, history_df in history_data.items():

    # One line per training method.
    plt.plot(
        history_df["step"],
        history_df["validation_ppl"],
        marker="o",
        label=method_name,
    )


plt.xlabel(
    "Training step"
)

plt.ylabel(
    "Validation perplexity"
)

plt.title(
    "QAT vs QAD vs QAH"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()


# Save the figure to /results.
plt.savefig(
    PLOT_OUTPUT,
    dpi=200,
)

plt.close()


# ---------------------------------------------------------
# Final output locations
# ---------------------------------------------------------

print("\n======================================")
print("Saved files")
print("======================================\n")

print(
    f"Summary table: "
    f"{SUMMARY_OUTPUT}"
)

print(
    f"Recovery percentages: "
    f"{RECOVERY_OUTPUT}"
)

print(
    f"Training curve plot: "
    f"{PLOT_OUTPUT}"
)
