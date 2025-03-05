from src.datasets.dataloader.dataset_2D import DataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    data_short_dir = 'data/kitti'  # Replace with your actual data directory
    datadir = os.path.join(root, data_short_dir)
    phase = 'debug'
    perturb_filenames = 'perturbation_neg_master_adjusted.csv'  # Replace with your actual file name
    batch_size = 1

    test_gen = DataGenerator(datadir=datadir,
                            phase=phase,
                            augmentation=None,
                            perturb_filenames=perturb_filenames,
                            loader='kitti_raw')
    num_cores = min(multiprocessing.cpu_count(), 64)
    test_dataloader = test_gen.create_data(batch_size=batch_size, shuffle=True, nthreads=num_cores)

    for batch in test_dataloader:
        left_img_batch = batch['left_img']  # batch of left image, id 02
        depth_batch = batch['depth']  # the corresponding depth ground truth of given id
        depth_neg = batch['depth_neg']

        # depth_batch has shape of (N, 1, H, W)
        depth_1 = depth_batch[0]  # (1, H, W)
        print(depth_batch.size())
        print(left_img_batch.size())

        # Permute
        lidar = depth_1.permute(1, 2, 0).numpy()  # position (H, W, 1) = (ca. 100x600x1)
        img_np1 = left_img_batch[0].permute(1, 2, 0).numpy()
        img_np2 = left_img_batch[0].permute(1, 2, 0).numpy()

        # Reshape tensor into (x, y, value)
        values_store = []
        lidar = np.squeeze(lidar)

        for (i, j), value in np.ndenumerate(lidar):
            values_store.append([j, i, value])

        values_store = np.array(values_store)
        values_store = np.delete(values_store, np.where(values_store[:, 2] == 0), axis=0)

        # negative sample
        depth_neg1 = depth_neg[0]
        lidar_neg = depth_neg1.permute(1, 2, 0).numpy()

        # Reshape tensor into (x, y, value)
        values_store_neg = []
        lidar_neg = np.squeeze(lidar_neg)

        for (i, j), value in np.ndenumerate(lidar_neg):
            values_store_neg.append([j, i, value])

        values_store_neg = np.array(values_store_neg)
        values_store_neg = np.delete(values_store_neg, np.where(values_store_neg[:, 2] == 0), axis=0)

        # negative sample
        depth_neg2 = depth_neg[0]
        lidar_neg2 = depth_neg2.permute(1, 2, 0).numpy()

        # Reshape tensor into (x, y, value)
        values_store_neg2 = []
        lidar_neg2 = np.squeeze(lidar_neg2)

        for (i, j), value in np.ndenumerate(lidar_neg2):
            values_store_neg2.append([j, i, value])

        values_store_neg2 = np.array(values_store_neg2)
        # values_store_neg = values_store_neg[::2]
        values_store_neg2 = np.delete(values_store_neg2, np.where(values_store_neg2[:, 2] == 0), axis=0)

        import matplotlib.pyplot as plt

        # First plot with img_np1 as the background
        plt.figure(figsize=(15, 7))
        plt.imshow(img_np1, alpha=1)
        # plt.scatter(values_store[:, 0], values_store[:, 1], c=values_store[:, 2], cmap='rainbow_r', alpha=0.5, s=3)
        plt.axis('off')  # Remove axis values
        plt.tight_layout()
        plt.show()

        # Second plot with only scatter and no background
        plt.figure(figsize=(15, 4.8))
        plt.imshow(img_np1, alpha=0.0)
        plt.scatter(values_store[:, 0], values_store[:, 1], c=values_store[:, 2], cmap='rainbow_r', alpha=1, s=3)
        plt.axis('off')  # Remove axis values
        plt.tight_layout()
        plt.show()

        #
        # plt.figure(figsize=(15, 7))
        # plt.imshow(img_np1)
        # plt.scatter(values_store_neg[:, 0], values_store_neg[:, 1], c=values_store_neg[:, 2], cmap='rainbow_r', alpha=0.5,
        #             s=5)
        # plt.tight_layout()
        # plt.show()