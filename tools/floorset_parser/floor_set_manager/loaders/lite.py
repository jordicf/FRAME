# Based on code from: https://github.com/IntelLabs/FloorSet

import os
import glob
import torch # type: ignore
from torch.utils.data import Dataset # type: ignore
from typing import Any
from tools.floorset_parser.floor_set_manager.utils.utils import is_dataset_downloaded, download_dataset


class FloorplanDatasetLite(Dataset):
    _test: bool

    def __init__(self, root:str, validation:bool = False):
        """
        Initializes the FloorplanDatasetLite.

        Args:
            root (str): The root directory where the Lite dataset is located.
            validation (bool, optional): If True, uses the validation/test Lite dataset. Defaults to False.
        """
        self._test = validation

        if self._test:
            folder_name = 'LiteTensorDataTest'
            if not is_dataset_downloaded(root, folder_name):            
                url = 'https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/LiteTensorDataTest.tar.gz'
                download_dataset(root, url)

            self.layouts_per_file = 1
            self.cached_file_idx = -1

            self.all_input_files = []
            self.all_label_files = []
            partition_range = range(21, 121) #number of partitions in prime
            identifier_range = range(1, 11)  # Identifiers from 1 to 10
            for worker_idx in partition_range:
                config_dir = os.path.join(root, f'{folder_name}/config_{worker_idx}')
                # Collect data files within the specified identifier range
                for identifier in identifier_range:
                    input_file_pattern = os.path.join(config_dir, f'litedata_{identifier}.pth')
                    label_file_pattern = os.path.join(config_dir, f'litelabel_{identifier}.pth')
                    if os.path.isfile(input_file_pattern):
                        self.all_input_files.append(input_file_pattern)
                    if os.path.isfile(label_file_pattern):
                        self.all_label_files.append(label_file_pattern)
        
        else:
            folder_name = 'floorset_lite'
            if not is_dataset_downloaded(root, folder_name):
                url = 'https://huggingface.co/datasets/IntelLabs/FloorSet/resolve/main/LiteTensorData_v2.tar.gz'
                download_dataset(root, url)

            self.layouts_per_file = 112
            self.cached_file_idx = -1

            self.all_files = []
            for worker_idx in range(100):
                self.all_files.extend(glob.glob(os.path.join(
                    root, folder_name, f"worker_{worker_idx}/layouts*")))


    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        if self._test:
            return len(self.all_input_files) * self.layouts_per_file
        else:
            return len(self.all_files) * self.layouts_per_file


    def __getitem__(self, idx:int)-> dict[torch.Tensor | Any]:
        """
        Retrieves a single data sample by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict[torch.Tensor | Any]: The data sample at the specified index.
        """
        if self._test:
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

        else:

            file_idx, layout_idx = divmod(idx, self.layouts_per_file)
            if file_idx != self.cached_file_idx:
                self.cached_input_file_contents = torch.load(self.all_files[file_idx])
                self.cached_file_idx = file_idx
                self.cached_layout_idx = layout_idx

            area_target = self.cached_input_file_contents[0][layout_idx][:,0]
            placement_constraints = self.cached_input_file_contents[0][layout_idx][:,1:]
            b2b_connectivity = self.cached_input_file_contents[1][layout_idx]
            p2b_connectivity = self.cached_input_file_contents[2][layout_idx]
            pins_pos = self.cached_input_file_contents[3][layout_idx]
            tree_sol = self.cached_input_file_contents[4][layout_idx]
            fp_sol = self.cached_input_file_contents[5][layout_idx]
            metrics_sol = self.cached_input_file_contents[6][layout_idx]

            input_data = (area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints)
            label_data = (tree_sol, fp_sol, metrics_sol)
            sample = {'input': input_data, 'label': label_data}
            return sample
    