import os
import shutil
import random
from pathlib import Path

# Define the directory where the script is located
script_dir = Path(__file__).resolve().parent

# Define the paths to the good and bad folders
good_folder = script_dir / 'good'
bad_folder = script_dir / 'bad'

# Define the path to the copied folder
copied_folder = Path('F:/Latest_MACKEREL_DATASETS/copied')

# Delete old copied folder if it exists
if copied_folder.exists():
    shutil.rmtree(copied_folder)

# Function to copy N random samples from source_folder to destination_folder
def copy_random_samples(source_folder, destination_folder, N):
    files = os.listdir(source_folder)
    selected_files = random.sample(files, min(N, len(files)))
    for file in selected_files:
        source_path = source_folder / file
        destination_path = destination_folder / file
        shutil.copyfile(source_path, destination_path)

# Take input for the number of random samples to copy
N = int(input("Enter the number of random samples to copy: "))

# Create good and bad folders inside the copied folder
copied_good_folder = copied_folder / 'good'
copied_bad_folder = copied_folder / 'bad'
copied_good_folder.mkdir(parents=True, exist_ok=True)
copied_bad_folder.mkdir(parents=True, exist_ok=True)

# Copy random samples from good and bad folders to respective folders inside copied folder
copy_random_samples(good_folder, copied_good_folder, N)
copy_random_samples(bad_folder, copied_bad_folder, N)

print(f"{N} random samples from both good and bad folders have been copied to the 'copied' folder.")
