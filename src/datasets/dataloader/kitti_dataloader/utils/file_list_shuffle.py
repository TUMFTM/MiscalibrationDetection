import os.path
import random

# Read the lines from the input file
folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
rel_path = 'filenames/eigen_train_val_large_files.txt'
full_filename = os.path.join(folder_path, rel_path)

with open(full_filename, 'r') as f:
    lines = f.readlines()

# Shuffle the lines
random.shuffle(lines)

# Write the shuffled lines to the output file
with open(full_filename, 'w') as f:
    f.writelines(lines)