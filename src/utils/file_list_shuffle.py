import os.path
import random

# Read the lines from the input file
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
rel_path = 'src/datasets/dataloader/kitti_dataloader/filenames/eigen_test_files.txt'
full_filename = os.path.join(project_path, rel_path)

with open(full_filename, 'r') as f:
    lines = f.readlines()

# Shuffle the lines
random.shuffle(lines)

# Write the shuffled lines to the output file
with open(full_filename, 'w') as f:
    f.writelines(lines)