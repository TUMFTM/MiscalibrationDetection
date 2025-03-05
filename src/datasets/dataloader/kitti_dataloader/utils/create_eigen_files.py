import os

def correction_existence(to_check_path, checked_file_path, removed_file_path):
    with open(to_check_path, 'r') as f:
        lines = f.readlines()

    with open(checked_file_path, 'w') as output_file, open(removed_file_path, 'w') as log_file:
        for line in lines:
            files = line.strip().split()[:-1]
            all_exist = True
            for file_path in files:
                full_path = os.path.join(root, data_folder, file_path)
                if not os.path.exists(full_path):
                    all_exist = False
                    break
            
            if all_exist:
                output_file.write(line)
            else:
                log_file.write(line)
        
    print(f"Results have been saved to {checked_file_path}")
    print(f"Removed lines have been logged to {removed_file_path}")
            

def create_eigen_files(sync_list_path, output_file_path):

    # Read the folder paths from the folder list file
    with open(sync_list_path, 'r') as file:
        syncs = [line.strip() for line in file.readlines()]

    # Open the output file in write mode
    with open(output_file_path, 'w') as output_file:
        for sync in syncs:

            date_folder = sync.split('_drive_')[0]
            sync_name = sync + '_sync'
            image02_path = os.path.join(root, "data/kitti", date_folder, sync_name, 'image_02/data')
            velo_path = os.path.join(root, "data/kitti", date_folder, sync_name, 'velodyne_points/data')

            # Check if the folder path is valid
            if os.path.isdir(image02_path):
                # Get the list of files in the current folder
                images_02 = os.listdir(image02_path)

                for image in images_02:
                    # Get file number
                    number_parts = image.split('.')[0]

                    # Formatting file name
                    image02_name = os.path.join(date_folder, sync_name, r'image_02/data', f'{number_parts}.png')
                    image03_name = os.path.join(date_folder, sync_name, r'image_03/data', f'{number_parts}.png')
                    lidar_name = os.path.join(date_folder, sync_name, r'velodyne_points/data', f'{number_parts}.bin')
                    perturbation_name = sync_name + '_' + number_parts

                    image02_name = image02_name.replace('\\', '/')
                    image03_name = image03_name.replace('\\', '/')
                    lidar_name = lidar_name.replace('\\', '/')

                    # Write the folder name (optional, for clarity)
                    output_file.write(f"{image02_name} {image03_name} {date_folder} {lidar_name} {perturbation_name}\n")

            else:
                output_file.write(f"Folder: {image02_path} (Invalid path)\n\n")

    print(f"File names have been written to {output_file_path}")


if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    root = os.path.abspath(os.path.join(current_file_path, '../../../../../../'))

    tag = 'test_unseen'

    # sync_list_path = os.path.join(root, f'src/datasets/dataloader/kitti_dataloader/utils/list_syncs/list_sync_{tag}.txt')
    # output_file_path = os.path.join(root, f'src/datasets/dataloader/kitti_dataloader/utils/log_files/unchecked_eigen_{tag}_files.txt')
    # checked_file_path = os.path.join(root, f'src/datasets/dataloader/kitti_dataloader/filenames/eigen_{tag}_files.txt')
    # removed_file_path = os.path.join(root, f'src/datasets/dataloader/kitti_dataloader/utils/log_files/removed_eigen_{tag}_files.txt')

    sync_list_path = os.path.join(root, f'src/datasets/dataloader/kitti_dataloader/utils/list_syncs/list_sync_{tag}.txt')
    output_file_path = os.path.join(root, f'outputs/others/unchecked_eigen_{tag}_files.txt')
    checked_file_path = os.path.join(root, f'outputs/others/eigen_{tag}_files.txt')
    removed_file_path = os.path.join(root, f'outputs/others/removed_eigen_{tag}_files.txt')

    data_folder = 'data/kitti'
    
    create_eigen_files(sync_list_path=sync_list_path, output_file_path=output_file_path)

    correction_existence(to_check_path=output_file_path, checked_file_path=checked_file_path, removed_file_path=removed_file_path)

    tag = 'test_unseen'

    # sync_list_path = os.path.join(root, f'src/datasets/dataloader/kitti_dataloader/utils/list_syncs/list_sync_{tag}.txt')
    # output_file_path = os.path.join(root, f'src/datasets/dataloader/kitti_dataloader/utils/log_files/unchecked_eigen_{tag}_files.txt')
    # checked_file_path = os.path.join(root, f'src/datasets/dataloader/kitti_dataloader/filenames/eigen_{tag}_files.txt')
    # removed_file_path = os.path.join(root, f'src/datasets/dataloader/kitti_dataloader/utils/log_files/removed_eigen_{tag}_files.txt')

    sync_list_path = os.path.join(root,
                                  f'src/datasets/dataloader/kitti_dataloader/utils/list_syncs/list_sync_{tag}.txt')
    output_file_path = os.path.join(root, f'outputs/others/unchecked_eigen_{tag}_files.txt')
    checked_file_path = os.path.join(root, f'outputs/others/eigen_{tag}_files.txt')
    removed_file_path = os.path.join(root, f'outputs/others/removed_eigen_{tag}_files.txt')

    create_eigen_files(sync_list_path=sync_list_path, output_file_path=output_file_path)

    correction_existence(to_check_path=output_file_path, checked_file_path=checked_file_path,
                         removed_file_path=removed_file_path)