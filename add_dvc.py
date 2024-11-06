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

# Function to run a command and ensure it completes successfully
def run_command(command):
    try:
        result = subprocess.run(command, check=True)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {' '.join(command)}")
        print(f"Error message: {e}")
        exit(1)  # Exit the script if any command fails

# List to collect all add commands
dvc_add_commands = []

# Loop over each dataset and add train, val, test, and model files to DVC
for dataset_name in dataset_names:
    # Data directories
    train_dir = os.path.join(dataset_root, dataset_name, "train")
    val_dir = os.path.join(dataset_root, dataset_name, "val")
    test_dir = os.path.join(dataset_root, dataset_name, "test")

    # Model file paths (e.g., results/dataset_name/)
    best_model_path = os.path.join(results_root, dataset_name, "best_model.h5")
    best_model_tuned_path = os.path.join(results_root, dataset_name, "best_model_tuned.h5")

    # Add train, val, and test directories to DVC (store commands to run later)
    if os.path.exists(train_dir):
        dvc_add_commands.append(["dvc", "add", train_dir])
    if os.path.exists(val_dir):
        dvc_add_commands.append(["dvc", "add", val_dir])
    if os.path.exists(test_dir):
        dvc_add_commands.append(["dvc", "add", test_dir])

    # Add model files to DVC if they exist (store commands to run later)
    # if os.path.exists(best_model_path):
    #     dvc_add_commands.append(["dvc", "add", best_model_path])
    # if os.path.exists(best_model_tuned_path):
    #     dvc_add_commands.append(["dvc", "add", best_model_tuned_path])

# Execute all dvc add commands and ensure they are completed
for command in dvc_add_commands:
    run_command(command)

# Now that all data has been added, push the changes to the remote storage
run_command(["dvc", "push"])

# Stage all changes (including untracked files and modifications)
run_command(["git", "add", "*.dvc", "*.py", "*.yaml"])

# Construct the commit message with timestamp
commit_message = f"Add datasets and model files to DVC - {timestamp}"

# Git commit with the message
run_command(["git", "commit", "-m", commit_message])

# Push the changes to the Git remote repository
run_command(["git", "push"])

print(f"All datasets and model files have been added, pushed to DVC, and committed to Git with message: '{commit_message}'")
