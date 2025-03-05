import csv
import os

def extract_names_to_txt(csv_file_path, txt_file_path):
    """
    Extracts the 'name' column from a CSV file and writes each name to a new line in a text file.

    Args:
        csv_file_path (str): Path to the input CSV file.
        txt_file_path (str): Path to the output text file.
    """
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        names = [row['name'] for row in csv_reader]

    with open(txt_file_path, 'w') as txt_file:
        txt_file.write('\n'.join(names))

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    root = os.path.abspath(os.path.join(current_file_path, '../../../../../../'))
    # Usage example
    csv_file_path =  os.path.join(root,'data/kitti_odom/perturbation_test_pos.csv')
    txt_file_path =  os.path.join(root,'data/kitti_odom/sequence_list_test.txt')
    extract_names_to_txt(csv_file_path, txt_file_path)
