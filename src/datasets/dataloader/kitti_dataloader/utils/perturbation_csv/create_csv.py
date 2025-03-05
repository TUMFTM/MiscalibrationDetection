import csv
import random
import os
import numpy as np

def choose_num_errors(theta_rad1, theta_rad2, theta_rad3, x, y, z, max_out):
    # Collect all errors in a list
    errors = [theta_rad1, theta_rad2, theta_rad3, x, y, z]
    num_zero_out = np.random.choice(range(0, max_out + 1))
    # Randomly choose which errors to zero out
    zero_out_indices = np.random.choice(range(len(errors)), num_zero_out, replace=False)

    for index in zero_out_indices:
        errors[index] = 0

    num_errors = len(errors) - num_zero_out
    errors.append(num_errors)
    return errors

def choose_num_errors_3(a, b, c, max_out):
    # Collect all errors in a list
    errors = [a, b, c]
    num_zero_out = np.random.choice(range(0, max_out + 1))
    # Randomly choose which errors to zero out
    zero_out_indices = np.random.choice(range(len(errors)), num_zero_out, replace=False)

    for index in zero_out_indices:
        errors[index] = 0

    num_errors = len(errors)-num_zero_out
    errors.append(num_errors)

    return errors

def generate_random_value(value_range):
    return round(random.uniform(*value_range)*np.random.choice([-1, 1]), 2)

# Define the ranges for the values
# # Pertubation negativ (labeled as wrong)
# x_range = (0.02, 0.1)
# y_range = (0.02, 0.1)
# z_range = (0.02, 0.1)
# range_rad1 = (0.5, 5)
# range_rad2 = (0.5, 5)
# range_rad3 = (0.5, 5)
# max_out = 5
# tag = 'neg_master'

# # # Perturbation positive
# x_range = (0, 0.02)
# y_range = (0, 0.02)
# z_range = (0, 0.02)
# range_rad1 = (0, 0.5)
# range_rad2 = (0, 0.5)
# range_rad3 = (0, 0.5)
# max_out = 6
# tag = 'pos_master'

# Define the ranges for the values
# Pertubation negativ (labeled as wrong)
x_range = (0.04, 0.1)
y_range = (0.04, 0.1)
z_range = (0.04, 0.1)
range_rad1 = (0.5, 5)
range_rad2 = (0.5, 5)
range_rad3 = (0.5, 5)
max_out = 5
tag = 'neg_master_adjusted_seen'

# # # Perturbation positive
# x_range = (0, 0.02)
# y_range = (0, 0.02)
# z_range = (0, 0.02)
# range_rad1 = (0, 0.3)
# range_rad2 = (0, 0.3)
# range_rad3 = (0, 0.3)
# max_out = 6
# tag = 'pos_master_adjusted'

# # Pertubation negativ (labeled as wrong)
# x_range = (0.1, 0.2)
# y_range = (0.1, 0.2)
# z_range = (0.1, 0.2)
# range_rad1 = (5.0, 10.0)
# range_rad2 = (5.0, 10.0)
# range_rad3 = (5.0, 10.0)
# max_out = 5
# tag = 'neg_ood'

# # Perturbation positive: noise
# x_range = (0, 0.005)
# y_range = (0, 0.005)
# z_range = (0, 0.005)
# range_rad1 = (0, 0.1)
# range_rad2 = (0, 0.1)
# range_rad3 = (0, 0.1)
# max_out = 6
# tag = 'noise'

# # # Translation only
# x_range = (0.1, 0.2)
# y_range = (0.1, 0.2)
# z_range = (0.1, 0.2)
# range_rad1 = (0, 0)
# range_rad2 = (0, 0)
# range_rad3 = (0, 0)
# max_out = 2
# tag = 'trans_compare_wei_seen'

# # Translation only
# x_range = (0.04, 0.1)
# y_range = (0.04, 0.1)
# z_range = (0.04, 0.1)
# range_rad1 = (0, 0)
# range_rad2 = (0, 0)
# range_rad3 = (0, 0)
# max_out = 2
# tag = 'trans_hard_seen'

# # Rotation only
# x_range = (0, 0)
# y_range = (0, 0)
# z_range = (0, 0)
# range_rad1 = (0.5, 1)
# range_rad2 = (0.5, 1)
# range_rad3 = (0.5, 1)
# max_out = 2
# tag = 'rot_only_seen'

# Define file paths
current_file_path = os.path.abspath(__file__)
root = os.path.abspath(os.path.join(current_file_path, '../../../../../../..'))

input_file_paths = [
    os.path.join(root, 'src/datasets/dataloader/kitti_dataloader/filenames/eigen_train_files.txt'),
    os.path.join(root, 'src/datasets/dataloader/kitti_dataloader/filenames/eigen_val_files.txt'),
    os.path.join(root, 'src/datasets/dataloader/kitti_dataloader/filenames/eigen_test_files.txt')
]

output_csv_file_path = os.path.join(root, f'src/datasets/dataloader/kitti_dataloader/utils/perturbation_csv/perturbation_{tag}.csv')

# Create the CSV file
with open(output_csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['name', 'x', 'y', 'z', 'theta_rad1', 'theta_rad2', 'theta_rad3', 'n'])

    # Loop over each input file path
    for input_file_path in input_file_paths:
        # Create the name list from the text file
        name_list = []
        with open(input_file_path, 'r') as f:
            file_names = [line.strip() for line in f.readlines()]
            for file_name in file_names:
                name = file_name.split('.bin ')[-1]
                name_list.append(name)

        # Process each name and generate random values
        for name in name_list:
            x = generate_random_value(x_range)
            y = generate_random_value(y_range)
            z = generate_random_value(z_range)
            theta_rad1 = generate_random_value(range_rad1)
            theta_rad2 = generate_random_value(range_rad2)
            theta_rad3 = generate_random_value(range_rad3)

            # Apply the error generation function
            theta_rad1, theta_rad2, theta_rad3, x, y, z, n = choose_num_errors(theta_rad1, theta_rad2, theta_rad3, x, y, z, max_out=max_out)

            # Write the row to the CSV file
            row = [
                name, x, y, z, theta_rad1, theta_rad2, theta_rad3, n
            ]
            writer.writerow(row)

print("CSV file created successfully with data from multiple input file paths.")
