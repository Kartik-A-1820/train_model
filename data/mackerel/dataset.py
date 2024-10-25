import os
import random
import shutil
from tqdm import tqdm

def split_directory(input_directory, output_directory):
    # Create the train, val, and test directories
    train_dir = os.path.join(output_directory, "train")
    val_dir = os.path.join(output_directory, "val")
    test_dir = os.path.join(output_directory, "test")

    # Create "good" and "bad" directories inside train, val, and test directories
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(directory, "good"), exist_ok=True)
        os.makedirs(os.path.join(directory, "bad"), exist_ok=True)

    # List the files in the "good" and "bad" folders
    good_files = os.listdir(os.path.join(input_directory, "good"))
    bad_files = os.listdir(os.path.join(input_directory, "bad"))

    # Shuffle the file lists randomly
    random.shuffle(good_files)
    random.shuffle(bad_files)

    # Calculate the split sizes for train, val, and test for good files
    total_good_files = len(good_files)
    good_train_split = int(0.8 * total_good_files)
    good_val_split = int(0.1 * total_good_files)

    # Calculate the split sizes for train, val, and test for bad files
    total_bad_files = len(bad_files)
    bad_train_split = int(0.8 * total_bad_files)
    bad_val_split = int(0.1 * total_bad_files)

    # Copy good files to the respective directories with progress bar
    for file in tqdm(good_files[:good_train_split], desc="Copying good files to train"):
        shutil.copy(os.path.join(input_directory, "good", file), os.path.join(train_dir, "good", file))
    for file in tqdm(good_files[good_train_split:good_train_split + good_val_split], desc="Copying good files to val"):
        shutil.copy(os.path.join(input_directory, "good", file), os.path.join(val_dir, "good", file))
    for file in tqdm(good_files[good_train_split + good_val_split:], desc="Copying good files to test"):
        shutil.copy(os.path.join(input_directory, "good", file), os.path.join(test_dir, "good", file))

    # Copy bad files to the respective directories with progress bar
    for file in tqdm(bad_files[:bad_train_split], desc="Copying bad files to train"):
        shutil.copy(os.path.join(input_directory, "bad", file), os.path.join(train_dir, "bad", file))
    for file in tqdm(bad_files[bad_train_split:bad_train_split + bad_val_split], desc="Copying bad files to val"):
        shutil.copy(os.path.join(input_directory, "bad", file), os.path.join(val_dir, "bad", file))
    for file in tqdm(bad_files[bad_train_split + bad_val_split:], desc="Copying bad files to test"):
        shutil.copy(os.path.join(input_directory, "bad", file), os.path.join(test_dir, "bad", file))

    # Remove the original "good" and "bad" directories
    shutil.rmtree(os.path.join(input_directory, "good"))
    shutil.rmtree(os.path.join(input_directory, "bad"))

if __name__ == "__main__":
    # Get the script's current directory as both input and output directories
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Run the split_directory function with the script's directory
    split_directory(script_directory, script_directory)
    
    print("Directory split completed.")
