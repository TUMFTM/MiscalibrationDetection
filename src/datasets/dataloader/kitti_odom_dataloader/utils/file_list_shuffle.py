import os.path
import random


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    root = os.path.abspath(os.path.join(current_file_path, '../../../../../../'))
    # Read the lines from the input file
    rel_path = 'data/kitti_odom/sequence_list_train.txt'
    full_filename = os.path.join(root, rel_path)

    with open(full_filename, 'r') as f:
        lines = f.readlines()

    # Shuffle the lines
    random.shuffle(lines)

    # Write the shuffled lines to the output file
    with open(full_filename, 'w') as f:
        f.writelines(lines)