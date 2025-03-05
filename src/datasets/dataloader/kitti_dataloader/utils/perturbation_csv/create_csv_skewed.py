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
x_range = (0.02, 0.1)
y_range = (0.02, 0.1)
z_range = (0.02, 0.1)
range_rad1 = (0.5, 5)
range_rad2 = (0.5, 5)
range_rad3 = (0.5, 5)
max_out = 5
tag = 'neg_master'

# Define file paths
current_file_path = os.path.abspath(__file__)
root = os.path.abspath(os.path.join(current_file_path, '../../../../../../..'))

input_file_path = os.path.join(root,'src/datasets/dataloader/kitti_dataloader/filenames/eigen_all_files.txt')
output_csv_file_path = os.path.join(root, f'src/datasets/dataloader/kitti_dataloader/filenames_generator/perturbation_csv/perturbation_{tag}.csv')

# Create the name list from the text file
name_list = []

with open(input_file_path, 'r') as file:
    file_names = [line.strip() for line in file.readlines()]
    for file_name in file_names:
        name = file_name.split('.bin ')[-1]
        name_list.append(name)


# Create the CSV file
with (open(output_csv_file_path, mode='w', newline='') as file):
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['name', 'x', 'y', 'z', 'theta_rad1', 'theta_rad2', 'theta_rad3', 'n'])

    # Write the data
    for name in name_list:
        x = generate_random_value(x_range)
        y = generate_random_value(y_range)
        z = generate_random_value(z_range)
        theta_rad1 = generate_random_value(range_rad1)
        theta_rad2 = generate_random_value(range_rad2)
        theta_rad3 = generate_random_value(range_rad3)

        theta_rad1, theta_rad2, theta_rad3, x, y, z, n = choose_num_errors(theta_rad1, theta_rad2, theta_rad3, x, y, z, max_out=max_out)

        # x, y, z, n = choose_num_errors_3(x, y, z, max_out=max_out)

        row = [
            name, x, y, z, theta_rad1, theta_rad2, theta_rad3, n
        ]
        writer.writerow(row)

print("Name list saved and CSV file created successfully.")
