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

def choose_num_errors_intr(theta_rad1, theta_rad2, theta_rad3, x, gamma, max_out):
    # Collect all errors in a list
    errors = [theta_rad1, theta_rad2, theta_rad3, x, gamma]
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


def generate_random_value_intr(value_range):
    return round(random.uniform(*value_range), 2)

def create_perturb_csv(sync_list_path, output_file_path, logs_path, range_dict):
    # Read the folder paths from the folder list file
    with open(sync_list_path, 'r') as file:
        names = [line.strip() for line in file.readlines()]

    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['name', 'x', 'y', 'z', 'theta_rad1', 'theta_rad2', 'theta_rad3', 'n'])

        for name in names:
            x = generate_random_value(range_dict["x_range"])
            y = generate_random_value(range_dict["y_range"])
            z = generate_random_value(range_dict["z_range"])
            theta_rad1 = generate_random_value(range_dict["range_rad1"])
            theta_rad2 = generate_random_value(range_dict["range_rad2"])
            theta_rad3 = generate_random_value(range_dict["range_rad3"])

            # Apply the error generation function
            theta_rad1, theta_rad2, theta_rad3, x, y, z, n = choose_num_errors(theta_rad1, theta_rad2,
                                                                               theta_rad3, x, y, z,
                                                                               max_out=range_dict["max_out"])

            # Write the row to the CSV file
            row = [
                name, x, y, z, theta_rad1, theta_rad2, theta_rad3, n
            ]
            writer.writerow(row)

    print(f"File names have been written to {output_file_path}")

def create_perturb_csv_intr(sync_list_path, output_file_path, logs_path, range_dict):
    # Read the folder paths from the folder list file
    with open(sync_list_path, 'r') as file:
        names = [line.strip() for line in file.readlines()]

    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['name', 'fu', 'fv', 'cu', 'cv', 'gamma', 'n'])

        for name in names:
            sign_u = np.random.choice([-1, 1])
            sign_v = np.random.choice([-1, 1])
            fu = generate_random_value(range_dict["fu"])*sign_u
            fv = generate_random_value(range_dict["fv"])*sign_v
            cu = generate_random_value(range_dict["cu"])*sign_u
            cv = generate_random_value(range_dict["cv"])*sign_v
            gamma = generate_random_value_intr(range_dict["gamma"])*sign_u


            # Apply the error generation function
            fu, fv, cu, cv, gamma, n = choose_num_errors_intr(fu, fv, cu, cv, gamma, max_out=range_dict["max_out"])

            # Write the row to the CSV file
            row = [
                name, fu, fv, cu, cv, gamma, n
            ]
            writer.writerow(row)

    print(f"File names have been written to {output_file_path}")


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    root = os.path.abspath(os.path.join(current_file_path, '../../../../../../'))

    # Define the ranges for the values
    # # Pertubation negativ (labeled as wrong)
    # range_dict = {
    #     "x_range": (0.04, 0.1),
    #     "y_range": (0.04, 0.1),
    #     "z_range": (0.04, 0.1),
    #     "range_rad1": (0.5, 5),
    #     "range_rad2": (0.5, 5),
    #     "range_rad3": (0.5, 5),
    #     "max_out": 5
    # }
    # tag = 'train'
    #
    # sync_list_path = os.path.join(root, f'data/kitti_odom/sequence_list_{tag}.txt')
    # output_logs_path = os.path.join(root, f'outputs/others/logs_{tag}_neg_csv.txt')
    # output_file_path = os.path.join(root, f'outputs/others/perturbation_{tag}_neg.csv')
    #
    # create_perturb_csv(sync_list_path=sync_list_path, output_file_path=output_file_path, logs_path=output_logs_path,
    #                    range_dict=range_dict)
    #
    # range_dict = {
    #     "x_range": (0, 0.02),
    #     "y_range":(0, 0.02),
    #     "z_range": (0, 0.02),
    #     "range_rad1": (0, 0.3),
    #     "range_rad2": (0, 0.3),
    #     "range_rad3": (0, 0.3),
    #     "max_out": 6
    # }
    # tag = 'train'
    #
    # sync_list_path = os.path.join(root, f'data/kitti_odom/sequence_list_{tag}.txt')
    # output_logs_path = os.path.join(root, f'outputs/others/logs_{tag}_pos_csv.txt')
    # output_file_path = os.path.join(root, f'outputs/others/perturbation_{tag}_pos.csv')
    #
    # create_perturb_csv(sync_list_path=sync_list_path, output_file_path=output_file_path, logs_path=output_logs_path,
    #                    range_dict=range_dict)
    #
    # range_dict = {
    #     "x_range": (0.04, 0.1),
    #     "y_range": (0.04, 0.1),
    #     "z_range": (0.04, 0.1),
    #     "range_rad1": (0.5, 5),
    #     "range_rad2": (0.5, 5),
    #     "range_rad3": (0.5, 5),
    #     "max_out": 5
    # }
    # tag = 'val'
    #
    # sync_list_path = os.path.join(root, f'data/kitti_odom/sequence_list_{tag}.txt')
    # output_logs_path = os.path.join(root, f'outputs/others/logs_{tag}_neg_csv.txt')
    # output_file_path = os.path.join(root, f'outputs/others/perturbation_{tag}_neg.csv')
    #
    # create_perturb_csv(sync_list_path=sync_list_path, output_file_path=output_file_path, logs_path=output_logs_path,
    #                    range_dict=range_dict)
    #
    # range_dict = {
    #     "x_range": (0, 0.02),
    #     "y_range":(0, 0.02),
    #     "z_range": (0, 0.02),
    #     "range_rad1": (0, 0.3),
    #     "range_rad2": (0, 0.3),
    #     "range_rad3": (0, 0.3),
    #     "max_out": 6
    # }
    # tag = 'val'
    #
    # sync_list_path = os.path.join(root, f'data/kitti_odom/sequence_list_{tag}.txt')
    # output_logs_path = os.path.join(root, f'outputs/others/logs_{tag}_pos_csv.txt')
    # output_file_path = os.path.join(root, f'outputs/others/perturbation_{tag}_pos.csv')
    #
    # create_perturb_csv(sync_list_path=sync_list_path, output_file_path=output_file_path, logs_path=output_logs_path,
    #                    range_dict=range_dict)

    # range_dict = {
    #     "x_range": (0.04, 0.1),
    #     "y_range": (0.04, 0.1),
    #     "z_range": (0.04, 0.1),
    #     "range_rad1": (0.5, 5),
    #     "range_rad2": (0.5, 5),
    #     "range_rad3": (0.5, 5),
    #     "max_out": 5
    # }
    # tag = 'test'
    #
    # sync_list_path = os.path.join(root, f'data/kitti_odom/sequence_list_{tag}.txt')
    # output_logs_path = os.path.join(root, f'outputs/others/logs_{tag}_neg_csv.txt')
    # output_file_path = os.path.join(root, f'outputs/others/perturbation_{tag}_neg.csv')
    #
    # create_perturb_csv(sync_list_path=sync_list_path, output_file_path=output_file_path, logs_path=output_logs_path,
    #                    range_dict=range_dict)
    #
    # range_dict = {
    #     "x_range": (0, 0.02),
    #     "y_range":(0, 0.02),
    #     "z_range": (0, 0.02),
    #     "range_rad1": (0, 0.3),
    #     "range_rad2": (0, 0.3),
    #     "range_rad3": (0, 0.3),
    #     "max_out": 6
    # }
    # tag = 'test'
    #
    # sync_list_path = os.path.join(root, f'data/kitti_odom/sequence_list_{tag}.txt')
    # output_logs_path = os.path.join(root, f'outputs/others/logs_{tag}_neg_csv.txt')
    # output_file_path = os.path.join(root, f'outputs/others/perturbation_{tag}_pos.csv')
    #
    # create_perturb_csv(sync_list_path=sync_list_path, output_file_path=output_file_path, logs_path=output_logs_path,
    #                    range_dict=range_dict)
    #
    # range_dict = {
    #     "x_range": (0, 0),
    #     "y_range": (0, 0),
    #     "z_range": (0, 0),
    #     "range_rad1": (0.5, 1),
    #     "range_rad2": (0.5, 1),
    #     "range_rad3": (0.5, 1),
    #     "max_out": 2
    # }
    # tag = 'test_rot'
    #
    # sync_list_path = os.path.join(root, f'data/kitti_odom/sequence_list_test.txt')
    # output_logs_path = os.path.join(root, f'outputs/others/logs_{tag}_neg_csv.txt')
    # output_file_path = os.path.join(root, f'outputs/others/perturbation_{tag}.csv')
    #
    # create_perturb_csv(sync_list_path=sync_list_path, output_file_path=output_file_path, logs_path=output_logs_path,
    #                    range_dict=range_dict)
    #
    # range_dict = {
    #     "x_range": (0.1, 0.2),
    #     "y_range": (0.1, 0.2),
    #     "z_range": (0.1, 0.2),
    #     "range_rad1": (0, 0),
    #     "range_rad2": (0, 0),
    #     "range_rad3": (0, 0),
    #     "max_out": 2
    # }
    # tag = 'test_trans'
    #
    # sync_list_path = os.path.join(root, f'data/kitti_odom/sequence_list_test.txt')
    # output_logs_path = os.path.join(root, f'outputs/others/logs_{tag}_neg_csv.txt')
    # output_file_path = os.path.join(root, f'outputs/others/perturbation_{tag}.csv')
    #
    # create_perturb_csv(sync_list_path=sync_list_path, output_file_path=output_file_path, logs_path=output_logs_path,
    #                    range_dict=range_dict)
    #
    # range_dict = {
    #     "x_range": (0, 0.005),
    #     "y_range":(0, 0.005),
    #     "z_range": (0, 0.005),
    #     "range_rad1": (0, 0.1),
    #     "range_rad2": (0, 0.1),
    #     "range_rad3": (0, 0.1),
    #     "max_out": 6
    # }
    # tag = 'noise'
    #
    # sync_list_path = os.path.join(root, f'data/kitti_odom/sequence_list_test.txt')
    # output_logs_path = os.path.join(root, f'outputs/others/logs_{tag}_neg_csv.txt')
    # output_file_path = os.path.join(root, f'outputs/others/perturbation_{tag}.csv')
    #
    # create_perturb_csv(sync_list_path=sync_list_path, output_file_path=output_file_path, logs_path=output_logs_path,
    #                    range_dict=range_dict)

    # range_dict = {
    #     "x_range": (0.1, 0.2),
    #     "y_range": (0.1, 0.2),
    #     "z_range": (0.1, 0.2),
    #     "range_rad1": (0.5, 1),
    #     "range_rad2": (0.5, 1),
    #     "range_rad3": (0.5, 1),
    #     "max_out": 5
    # }
    # tag = 'test_wei_scale_1'

    # range_dict = {
    #     "x_range": (0.1, 0.2),
    #     "y_range": (0.1, 0.2),
    #     "z_range": (0.1, 0.2),
    #     "range_rad1": (5, 10),
    #     "range_rad2": (5, 10),
    #     "range_rad3": (5, 10),
    #     "max_out": 5
    # }
    # tag = 'test_unseen'

    # range_dict = {
    #     "x_range": (0.04, 0.1),
    #     "y_range": (0.04, 0.1),
    #     "z_range": (0.04, 0.1),
    #     "range_rad1": (0, 0),
    #     "range_rad2": (0, 0),
    #     "range_rad3": (0, 0),
    #     "max_out": 2
    # }
    # tag = 'test_trans_hard'

    # range_dict = {
    #     "x_range": (0, 0),
    #     "y_range": (0, 0),
    #     "z_range": (0, 0),
    #     "range_rad1": (1, 5),
    #     "range_rad2": (1, 5),
    #     "range_rad3": (1, 5),
    #     "max_out": 2
    # }
    # tag = 'test_rot_easy'

    # create_perturb_csv(sync_list_path=sync_list_path, output_file_path=output_file_path, logs_path=output_logs_path,
    #                    range_dict=range_dict)
    
    # range_dict = {
    #     "fu": (1, 5),
    #     "fv": (1, 5),
    #     "cu": (1, 5),
    #     "cv": (1, 5),
    #     "max_out": 3
    # }
    # tag = 'test_intrinsic'

    # range_dict = {
    #     "fu": (3, 5),
    #     "fv": (3, 5),
    #     "cu": (3, 5),
    #     "cv": (3, 5),
    #     "gamma": (0.3, 0.5),
    #     "max_out": 4
    # }
    # tag = 'test_intrinsic_abs'

    range_dict = {
        "fu": (10, 20),
        "fv": (10, 20),
        "cu": (10, 20),
        "cv": (10, 20),
        "gamma": (10, 20),
        "max_out": 4
    }
    tag = 'test_intrinsic_10_20'
        
    # range_dict = {
    #     "fu": (10, 20),
    #     "fv": (10, 20),
    #     "cu": (10, 20),
    #     "cv": (10, 20),
    #     "max_out": 3
    # }
    # tag = 'test_intrinsic_very_large'

    sync_list_path = os.path.join(root, f'data/kitti_odom/sequence_list_test.txt')
    output_logs_path = os.path.join(root, f'data/kitti_odom/logs_{tag}_neg_csv.txt')
    output_file_path = os.path.join(root, f'data/kitti_odom/perturbation_{tag}.csv')

    create_perturb_csv_intr(sync_list_path=sync_list_path, output_file_path=output_file_path, logs_path=output_logs_path,
                       range_dict=range_dict)
