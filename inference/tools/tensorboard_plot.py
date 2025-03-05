import os
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np


def calculate_iqr_limits(values):
    """
    Calculate the lower and upper limits based on IQR to clip the plot,
    ensuring the lower bound is not lower than the minimum data value.
    """
    if len(values) == 0:
        raise ValueError("No values available for IQR calculation.")

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    # Calculate IQR bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Ensure lower bound is not lower than the minimum value
    lower_bound = max(lower_bound, np.min(values))

    return lower_bound, upper_bound


def plot_tensorboard_logs(log_files, training_tag, validation_tag, plot_tag, enable_validation=True):
    """
    Plots both training and (optionally) validation losses from TensorBoard log files,
    clipping the y-axis based on IQR.

    Parameters:
    - log_files: List of absolute paths to TensorBoard log files.
    - training_tag: The scalar tag for training loss (e.g., 'training_loss_epoch').
    - validation_tag: The scalar tag for validation loss (e.g., 'validation_loss_epoch').
    - plot_tag: The title for the plot.
    - enable_validation: Boolean flag to enable or disable the validation plot.
    """
    plt.figure(figsize=(10, 6))

    all_values = []  # To collect all values for calculating the global IQR-based limits

    # Iterate through all log files provided
    for log_file in log_files:
        event_acc = event_accumulator.EventAccumulator(log_file)
        event_acc.Reload()

        folder_name = os.path.basename(os.path.dirname(log_file))
        # folder_name_cleaned = folder_name.replace('run', '').replace('contrastive', '').strip('_')

        if '_aug_' in folder_name:
            folder_name_cleaned = "Calib (II)"
        else:
            folder_name_cleaned = "Calib (I)"

        # Handle training loss
        if training_tag in event_acc.Tags()['scalars']:
            training_scalars = event_acc.Scalars(training_tag)
            training_steps = [scalar.step for scalar in training_scalars]
            training_values = [scalar.value for scalar in training_scalars]

            # Append training values for IQR calculation
            all_values.extend(training_values)

            # Plot training loss
            training_line, = plt.plot(training_steps, training_values, label=f'{folder_name_cleaned} - Training')
        else:
            print(f"Warning: Tag '{training_tag}' not found in {log_file}")

        # Handle validation loss only if enabled
        if enable_validation and validation_tag in event_acc.Tags()['scalars']:
            validation_scalars = event_acc.Scalars(validation_tag)
            validation_steps = [scalar.step for scalar in validation_scalars]
            validation_values = [scalar.value for scalar in validation_scalars]

            # Append validation values for IQR calculation
            all_values.extend(validation_values)

            # Plot validation loss with same color as training but dashed line
            plt.plot(validation_steps, validation_values, linestyle='--', color=training_line.get_color(),
                     label=f'{folder_name_cleaned} - Validation')
        elif enable_validation:
            print(f"Warning: Tag '{validation_tag}' not found in {log_file}")

    # Check if all_values contains any data before proceeding with IQR limits
    if len(all_values) == 0:
        raise ValueError(f"No data found for the tags '{training_tag}' or '{validation_tag}'.")

    # Calculate the IQR-based limits
    lower_bound, upper_bound = calculate_iqr_limits(all_values)

    # Set y-axis limits to clip outliers based on IQR
    plt.ylim(lower_bound, upper_bound)

    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title(f'{plot_tag}', fontsize=24)

    plt.tick_params(axis='both', labelsize=16)

    # Add grid to the plot
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Set legend with smaller font size
    plt.legend(loc='best', fontsize=14)

    plt.show()


def get_contrastive_log_files(root_dir, tag):
    """
    Recursively find all log files in folders containing '_contrastive' or '_classifier'.
    """
    log_files = []

    if tag == "classifier":
        dir_tag = "_classifier"
    else:
        dir_tag = "_contrastive"

    # Walk through all directories and subdirectories starting from root_dir
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            # Check if the folder name contains the relevant tag
            if dir_tag in dir_name:
                dir_path = os.path.join(root, dir_name)
                # Find all event files in this directory
                for file_name in os.listdir(dir_path):
                    if file_name.startswith('events.out.tfevents'):
                        log_files.append(os.path.join(dir_path, file_name))

    return log_files


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    root_dir = os.path.join(root, 'outputs/logs/active')
    tag = "contrastive"  # "classifier" or "contrastive

    # Get all log files from '_contrastive' or '_classifier' folders
    log_files = get_contrastive_log_files(root_dir, tag)
    print(log_files)

    # Define the tags and plot tag
    if tag == "classifier":
        training_tag = 'training_cls_loss_epoch'
        validation_tag = 'validation_cls_loss_epoch'
        plot_tag = 'BCE Loss - Classifier'
    else:
        training_tag = 'training_loss_epoch'
        validation_tag = 'validation_loss_epoch'
        plot_tag = 'Contrastive Loss'

    # Call the plot function with the validation plot disabled (set to False to disable)
    enable_validation = True  # Change this to False to disable validation plot

    plot_tensorboard_logs(log_files, training_tag, validation_tag, plot_tag, enable_validation)
