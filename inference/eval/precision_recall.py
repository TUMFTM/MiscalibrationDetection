from sklearn.metrics import auc
import torch
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.datasets.dataloader.dataset_2D import DataGenerator
from sklearn.metrics import precision_recall_curve
import multiprocessing
from sklearn.metrics import f1_score
import numpy as np
from collections import Counter
import time

def pr_auc(label, prediction):
    # Convert tensors to numpy arrays
    label = np.concatenate([tensor.cpu().numpy() for tensor in label])
    prediction = np.concatenate([tensor.cpu().numpy() for tensor in prediction])

    # Calculate precision-recall pairs for different thresholds
    precision, recall, thresholds = precision_recall_curve(label, prediction)

    # Calculate PR AUC
    pr_auc = auc(recall, precision)

    return {
        "Precision": precision,
        "Recall": recall,
        "thresholds": thresholds,
        "pr_auc": pr_auc
    }


def plot_pr_curve(pr_data, plot_file):
    plt.figure(figsize=(8, 6))
    plt.plot(pr_data['Recall'], pr_data['Precision'], marker='.', label=f'PR AUC = {pr_data["pr_auc"]:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_file)
    # plt.show()


def confusion_matrix(label, prediction, threshold=0.5, name_list=None):
    # Convert tensors to numpy arrays
    label = np.concatenate([tensor.cpu().numpy() for tensor in label])
    prediction = np.concatenate([tensor.cpu().numpy() for tensor in prediction])

    # Binarize predictions based on the threshold
    binarized_prediction = (prediction >= threshold).astype(int)

    # Calculate confusion matrix components
    TP = np.sum((label == 1) & (binarized_prediction == 1))
    TN = np.sum((label == 0) & (binarized_prediction == 0))
    FP = np.sum((label == 0) & (binarized_prediction == 1))
    FN = np.sum((label == 1) & (binarized_prediction == 0))

    # Extract FP and FN names if name_list is provided
    if name_list is not None:
        # Get indices of FP and FN
        fp_indices = np.where((label == 0) & (binarized_prediction == 1))[0]
        fn_indices = np.where((label == 1) & (binarized_prediction == 0))[0]

        # Extract names of FP and FN
        fp_names = [name_list[i] for i in fp_indices]
        fn_names = [name_list[i] for i in fn_indices]

        # Count occurrences of specific dates in FP and FN names
        date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        fp_dates = [name.split('_drive_')[0] for name in fp_names]  # Extract date parts from names
        fn_dates = [name.split('_drive_')[0] for name in fn_names]

        fp_date_counts = {date: fp_dates.count(date) for date in date_list}
        fn_date_counts = {date: fn_dates.count(date) for date in date_list}

        return TP, TN, FP, FN, fp_names, fn_names, fp_date_counts, fn_date_counts

    return TP, TN, FP, FN


def plot_distribution(label, prediction, dist_save):
    flipped_prediction = 1 - prediction # still for wrong CL Function
    flipped_label = 1 - label

    pred_label_1 = flipped_prediction[flipped_label == 1].cpu()
    pred_label_0 = flipped_prediction[flipped_label == 0].cpu()

    # Plot the distributions
    plt.figure(figsize=(10, 6))

    # Check if there are any label == 1 predictions to plot
    if len(pred_label_1) > 0:
        plt.hist(pred_label_1.numpy(), bins=30, alpha=0.4, label='Label == 1', color='red', density=True)
    else:
        print("No instances of label == 1")

    # Check if there are any label == 0 predictions to plot
    if len(pred_label_0) > 0:
        plt.hist(pred_label_0.numpy(), bins=30, alpha=0.4, label='Label == 0', color='blue', density=True)
    else:
        print("No instances of label == 0")

    # Adding titles and labels
    plt.title('Distribution of Predictions')
    plt.xlabel('Prediction Values')
    plt.ylabel('Density')
    plt.legend(fontsize=14)

    # Set the y-axis limit from 0 to 30
    plt.ylim(0, 30)

    # Save the plot
    plt.savefig(dist_save)
    plt.close()
    # Show the plot
    # plt.show()


def find_optimal_threshold(label, prediction):
    """
    Function to find the optimal threshold for maximizing the F1 score.

    Returns:
    max_f1_score (float): The highest F1 score achieved.
    opt_threshold (float): The threshold corresponding to the highest F1 score.
    """
    label_np = label.cpu().numpy()
    prediction_np = prediction.cpu().numpy()

    thresholds = np.arange(0.0, 1.0, 0.01)  # Thresholds from 0 to 1 in steps of 0.01
    f1_scores = []
    prediction_np = np.array(prediction_np)

    for threshold in thresholds:
        # Apply threshold to get predicted labels, ensuring it is a NumPy array
        y_pred = (prediction_np >= threshold).astype(int)
        f1 = f1_score(label_np, y_pred)
        f1_scores.append(f1)

    # Find the threshold with the maximum F1 score
    max_f1_score = max(f1_scores)
    opt_threshold = thresholds[np.argmax(f1_scores)]

    return max_f1_score, float(opt_threshold)


def pr_evaluation(device, data_root, output_dir, model_cls, perturbation_eval, mode='labeled', show_plot=False, loader='kitti_raw', intrinsic=False):

    batch_size = 64
    num_cores = min(multiprocessing.cpu_count(), 64)

    eval_gen = DataGenerator(data_root, 'test', perturb_filenames=perturbation_eval,
                             augmentation="perturbation_noise.csv", loader=loader, intrinsic=intrinsic)
    eval_dataloader = eval_gen.create_data(batch_size=batch_size, shuffle=False, nthreads=num_cores)

    os.makedirs(output_dir, exist_ok=True)
    fp_output_file = os.path.join(output_dir, f'output_fp.txt')
    fn_output_file = os.path.join(output_dir, f'output_fn.txt')
    fp_opt = os.path.join(output_dir, f'fp_opt.txt')
    fn_opt = os.path.join(output_dir, f'fn_opt.txt')
    results_file = os.path.join(output_dir, f'output_pr_auc.txt')
    prauc_file = os.path.join(output_dir, f'output_prauc.png')
    dist_file = os.path.join(output_dir, f'output_distribution.png')

    label = torch.empty(0, 1, device=device)
    prediction = torch.empty(0, 1, device=device)
    fp_list = []
    fn_list = []
    all_depth_names = []
    sum_TP = 0
    sum_TN = 0
    sum_FP = 0
    sum_FN = 0
    iter = 0

    # Initialize counters for date occurrences
    fp_date_total_counts = Counter()
    fn_date_total_counts = Counter()

    with torch.no_grad():
        print("Evaluation is started")
        start_time = time.time()
        for batch in eval_dataloader:
            left_img_batch = batch['left_img'].to(device)  # batch of left image, id 02
            depth_batch = batch['depth'].to(device)  # the corresponding depth ground truth of given id
            depth_neg = batch['depth_neg'].to(device)
            depth_name = batch['name']

            batch_length = len(depth_batch)

            if mode == 'labeled':
                label_val = torch.cat(
                    [torch.ones(batch_length, device=device), torch.zeros(batch_length, device=device)]).unsqueeze(1)
                left_img_batch = torch.cat((left_img_batch, left_img_batch), dim=0)
                stacked_depth_batch = torch.cat((depth_batch, depth_neg), dim=0)
            elif mode == 'random_paired':
                label_val = torch.cat(
                    [torch.ones(batch_length, device=device), torch.zeros(batch_length, device=device)]).unsqueeze(1)
                shuffled_depth_batch = depth_batch.clone()
                shuffled_depth_batch = shuffled_depth_batch[torch.randperm(shuffled_depth_batch.size(0))]
                left_img_batch = torch.cat((left_img_batch, left_img_batch), dim=0)
                stacked_depth_batch = torch.cat((depth_batch, shuffled_depth_batch), dim=0)
            elif mode == 'random_value':
                label_val = torch.cat(
                    [torch.ones(batch_length, device=device), torch.zeros(batch_length, device=device)]).unsqueeze(1)
                scaled_depth_batch = depth_batch.clone()
                non_zero_mask = scaled_depth_batch != 0
                non_zero_values = scaled_depth_batch[non_zero_mask]

                overall_min = non_zero_values.min().item()
                overall_max = non_zero_values.max().item()

                # Generate random values in the range [overall_min, overall_max] with the same shape as non-zero elements
                random_values = torch.rand_like(non_zero_values, device=device) * (
                        overall_max - overall_min) + overall_min

                # Replace the non-zero elements in the batch with the generated random values
                scaled_depth_batch[non_zero_mask] = random_values
                left_img_batch = torch.cat((left_img_batch, left_img_batch), dim=0)
                stacked_depth_batch = torch.cat((depth_batch, scaled_depth_batch), dim=0)

            N, C, H, W = left_img_batch.size()

            pred_cls = model_cls.forward(image=left_img_batch, lidar=stacked_depth_batch, H=H, W=W)

            # Concatenate the batch results to the full tensors
            label = torch.cat((label, label_val), dim=0)
            prediction = torch.cat((prediction, pred_cls), dim=0)
            all_depth_names.extend(depth_name + depth_name)

            TP, TN, FP, FN, fp_names, fn_names, fp_date_counts, fn_date_counts = confusion_matrix(label=label_val,
                                                                                                  prediction=pred_cls,
                                                                                                  name_list=(depth_name + depth_name),
                                                                                                  threshold=0.5)
            fp_list.extend(fp_names)
            fn_list.extend(fn_names)
            sum_TP += TP
            sum_TN += TN
            sum_FP += FP
            sum_FN += FN

            # Update date counts
            fp_date_total_counts.update(fp_date_counts)
            fn_date_total_counts.update(fn_date_counts)

            iter += 1
            print(f'Iteration {iter} finished')

        end_time = time.time()

        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.6f} seconds")

    print(f'True Positives: {sum_TP}')
    print(f'True Negatives: {sum_TN}')
    print(f'False Positives: {sum_FP}')
    print(f'False Negatives: {sum_FN}')

    accuracy = (sum_TP+sum_TN)/(sum_TP+sum_TN+sum_FP+sum_FN)
    precision = sum_TP / (sum_TP + sum_FP)
    recall = sum_TP / (sum_TP + sum_FN)

    # Save FP list to a text file
    with open(fp_output_file, 'w') as f:
        f.write("Time:%.6f\n" % inference_time)
        f.write("False Positives (FP):\n")
        for item in fp_list:
            f.write("%s\n" % item)

    # Save FN list to a text file
    with open(fn_output_file, 'w') as f:
        f.write("False Negatives (FN):\n")
        for item in fn_list:
            f.write("%s\n" % item)

    np.set_printoptions(threshold=np.inf)
    results = pr_auc(label=label, prediction=prediction)
    max_f1, opt_threshold = find_optimal_threshold(label=label, prediction=prediction)
    TP, TN, FP, FN, fp_names_opt, fn_names_opt, fp_date_counts_opt, fn_date_counts_opt = confusion_matrix(label=label, prediction=prediction,
                                                                                                          name_list=all_depth_names,
                                                                                                          threshold=opt_threshold)

    accuracy_opt = (TP+TN)/(TP+TN+FP+FN)
    precision_opt = TP / (TP + FP)
    recall_opt = TP / (TP + FN)

    # Save FP list to a text file
    with open(fp_opt, 'w') as f:
        f.write("False Positives (FP):\n")
        for item in fp_list:
            f.write("%s\n" % item)

    # Save FN list to a text file
    with open(fn_opt, 'w') as f:
        f.write("False Negatives (FN):\n")
        for item in fn_list:
            f.write("%s\n" % item)

    with open(results_file, 'w') as f:
        f.write("TP(0.5): %i\n" % sum_TP)
        f.write("TN(0.5): %i\n" % sum_TN)
        f.write("FP(0.5): %i\n" % sum_FP)
        f.write("FN(0.5): %i\n" % sum_FN)
        f.write("accuracy(0.5): %f\n" % accuracy)
        f.write("precision(0.5): %f\n" % precision)
        f.write("recall(0.5): %f\n" % recall)
        f.write("False Positive Date Counts:\n")
        for date, count in fp_date_total_counts.items():
            f.write(f"{date}: {count}\n")
        f.write("\n")
        f.write("False Negative Date Counts:\n")
        for date, count in fn_date_total_counts.items():
            f.write(f"{date}: {count}\n")
        f.write("============================================\n")
        f.write("Optimal Threshold:\n")
        f.write("TP(%f): %i\n" % (opt_threshold, TP))
        f.write("TN(%f): %i\n" % (opt_threshold, TN))
        f.write("FP(%f): %i\n" % (opt_threshold, FP))
        f.write("FN(%f): %i\n" % (opt_threshold, FN))
        f.write("accuracy(%f): %f\n" % (opt_threshold,accuracy_opt))
        f.write("precision(%f): %f\n" % (opt_threshold,precision_opt))
        f.write("recall(%f): %f\n" % (opt_threshold,recall_opt))
        f.write("============================================\n")
        f.write("Precision:\n")
        f.write("%s\n" % results["Precision"])
        f.write("Recall:\n")
        f.write("%s\n" % results["Recall"])
        f.write("Thresholds:\n")
        f.write("%s\n" % results["thresholds"])
        f.write("PR AUC:\n")
        f.write("%s\n" % results["pr_auc"])

    plot_distribution(label=label, prediction=prediction, dist_save=dist_file)
    plot_pr_curve(results, plot_file=prauc_file)

    print(f'results are saved in {output_dir}')

    return None
