import os
import subprocess
from datetime import datetime

# Path to your dataset root
dataset_root = "data"
results_root = "results"

# Get the current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Get the list of dataset directories
dataset_names = os.listdir(dataset_root)

# Loop over each dataset and add train, val, test, and model files to DVC
for dataset_name in dataset_names:
    # Data directories
    train_dir = os.path.join(dataset_root, dataset_name, "train")
    val_dir = os.path.join(dataset_root, dataset_name, "val")
    test_dir = os.path.join(dataset_root, dataset_name, "test")

    # Model file paths (e.g., results/dataset_name/)
    best_model_path = os.path.join(results_root, dataset_name, "best_model.h5")
    best_model_tuned_path = os.path.join(results_root, dataset_name, "best_model_tuned.h5")

    # Add train, val, and test directories to DVC
    if os.path.exists(train_dir):
        subprocess.run(["dvc", "add", train_dir], check=True)
    if os.path.exists(val_dir):
        subprocess.run(["dvc", "add", val_dir], check=True)
    if os.path.exists(test_dir):
        subprocess.run(["dvc", "add", test_dir], check=True)

    # Add model files to DVC if they exist
    if os.path.exists(best_model_path):
        subprocess.run(["dvc", "add", best_model_path], check=True)
    if os.path.exists(best_model_tuned_path):
        subprocess.run(["dvc", "add", best_model_tuned_path], check=True)

# DVC Push: Push all added files to the remote storage
subprocess.run(["dvc", "push"], check=True)

# Stage all changes (including untracked files and modifications)
subprocess.run(["git", "add", "-A"], check=True)

# Construct the commit message with timestamp
commit_message = f"Add datasets and model files to DVC - {timestamp}"

# Git commit with the message
subprocess.run(["git", "commit", "-m", commit_message], check=True)

# Push the changes to the Git remote repository
subprocess.run(["git", "push"], check=True)

print(f"All datasets and model files have been added, pushed to DVC, and committed to Git with message: '{commit_message}'")
