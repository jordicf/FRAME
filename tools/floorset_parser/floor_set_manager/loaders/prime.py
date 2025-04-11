# Based on code from: https://github.com/IntelLabs/FloorSet

import os
import torch
from torch.utils.data import Dataset
from typing import Any
from tools.floorset_parser.floor_set_manager.utils.utils import is_dataset_downloaded, download_dataset


class FloorplanDatasetPrime(Dataset):
    _test: bool


    def __init__(self, root: str, validation: bool = False):
        """
        Initializes the FloorplanDatasetPrime.

        Args:
            root (str): The root directory where the Prime dataset is located.
            validation (bool, optional): If True, uses the validation/test Prime dataset. Defaults to False.
        """
        self._test = validation
        
        if self._test:
            self.layouts_per_file = 1
            self.cached_file_idx = -1
            folder_name = 'PrimeTensorDataTest'
            if not is_dataset_downloaded(root, folder_name):
                url = 'https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/PrimeTensorDataTest.tar.gz'
                download_dataset(root, url)
        else:
            self.layouts_per_file = 1000
            self.cached_file_idx = -1
            folder_name = 'PrimeTensorData'
            if not is_dataset_downloaded(root, folder_name):
                url = 'https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/PrimeTensorData.tar.gz'
                download_dataset(root, url)

        self.all_input_files = []
        self.all_label_files = []
        partition_range = range(21, 121) #number of partitions in prime
        identifier_range = range(1, 11)  # Identifiers from 1 to 10
        for worker_idx in partition_range:
            config_dir = os.path.join(root, f'{folder_name}/config_{worker_idx}')
            # Collect data files within the specified identifier range
            for identifier in identifier_range:
                input_file_pattern = os.path.join(config_dir, f'primedata_{identifier}.pth')
                label_file_pattern = os.path.join(config_dir, f'primelabel_{identifier}.pth')
                if os.path.isfile(input_file_pattern):
                    self.all_input_files.append(input_file_pattern)
                if os.path.isfile(label_file_pattern):
                    self.all_label_files.append(label_file_pattern)  


    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.all_input_files) * self.layouts_per_file

    def __getitem__(self, idx:int)-> dict[torch.Tensor | Any]:
        """
        Retrieves a single data sample by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict[torch.Tensor | Any]: The data sample at the specified index.
        """
        file_idx, layout_idx = divmod(idx, self.layouts_per_file)
        if file_idx != self.cached_file_idx:
            self.cached_input_file_contents = torch.load(self.all_input_files[file_idx])
            self.cached_label_file_contents = torch.load(self.all_label_files[file_idx])
            self.cached_file_idx = file_idx
            
        area_target = self.cached_input_file_contents[layout_idx][0][:,0]
        placement_constraints = self.cached_input_file_contents[layout_idx][0][:,1:]
        b2b_connectivity = self.cached_input_file_contents[layout_idx][1]
        p2b_connectivity = self.cached_input_file_contents[layout_idx][2]
        pins_pos = self.cached_input_file_contents[layout_idx][3]
        fp_sol = self.cached_label_file_contents[layout_idx][1]
        metrics_sol = self.cached_label_file_contents[layout_idx][0]

        input_data = (area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints)
        label_data = (fp_sol, metrics_sol)
        sample = {'input': input_data, 'label': label_data}
        return sample
