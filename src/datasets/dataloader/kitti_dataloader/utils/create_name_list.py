import re
import os

def extract_name_from_line(line):
    # Extract parts of the name from the given line format
    match = re.match(r'(\d{4}_\d{2}_\d{2})/.+drive_(\d{4})_sync/.+/(\d{10})\.png', line)
    if match:
        date_part, drive_part, frame_part = match.groups()
        return f"{date_part}_{drive_part}_{frame_part}"
    return None

# Read the file and create the name list
def create_name_list_from_file(file_path):
    name_list = []
    with open(file_path, 'r') as file:
        for line in file:
            name = extract_name_from_line(line)
            if name:
                name_list.append(name)
    return name_list

# Define the file path
current_file_path = os.path.abspath(__file__)
root = os.path.abspath(os.path.join(current_file_path, '../../../../../../..'))
input_file_path = os.path.join(root, 'src/datasets/dataloader/kitti_dataloader/filenames_generator/eigen_train_files.txt')
output_file_path = os.path.join(root, 'src/dataset/dataloader/kitti_dataloader/filenames_generator/name_list_train.txt')

# Create the name list
name_list = create_name_list_from_file(input_file_path)

with open(output_file_path, 'w') as output_file:
    for name in name_list:
        output_file.write(f"{name}\n")