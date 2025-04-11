# Based on code from: https://github.com/IntelLabs/FloorSet

import torch
import torch.nn.functional as F


def floorplan_collate(batch):
    """Function to collate the datasets Prime (training and test) and Lite test"""
    area_target = [item['input'][0] for item in batch]
    b2b_connectivity = [item['input'][1] for item in batch]
    p2b_connectivity = [item['input'][2] for item in batch]
    pins_pos = [item['input'][3] for item in batch]
    placement_constraints = [item['input'][4] for item in batch]
    fp_sol = [item['label'][0] for item in batch]
    metrics_sol = [item['label'][1] for item in batch]


    def pad_polygons(sol):
        # Determine the maximum number of tensors in any list
        max_length = max(len(tensor_list) for tensor_list in sol)
        # Set target size for padding
        target_rows = 14
        target_cols = 2
        # List to store the padded tensors for each list
        all_group_padded_tensors = []
        # Iterate over each list of tensors in sol
        for tensor_list in sol:
            # List to store padded tensors within the current list
            padded_tensors = []
            # Pad each tensor to have target_rows
            for tensor in tensor_list:
                # Calculate the padding required
                pad_rows = target_rows - tensor.size(0)
                # Create the padding tuple (left, right, top, bottom)
                pad = (0, 0, 0, pad_rows)  # (left, right, top, bottom)
                # Pad the tensor using F.pad
                padded_tensor = F.pad(tensor, pad, value=-1)
                # Append the padded tensor to the list
                padded_tensors.append(padded_tensor)
            # If there are fewer tensors than max_length, pad the list
            while len(padded_tensors) < max_length:
                # Create a tensor of size (target_rows, target_cols) filled with -1
                empty_tensor = torch.full((target_rows, target_cols), -1)
                padded_tensors.append(empty_tensor)
            # Stack the padded tensors for the current list
            group_tensor = torch.stack(padded_tensors)
            # Append the group's tensor to the final list
            all_group_padded_tensors.append(group_tensor)
        # Stack all group tensors into a single tensor
        final_tensor = torch.stack(all_group_padded_tensors)
        return final_tensor
    

    def pad_inputs(tens_list):
        ndims = tens_list[0].ndim
        max_dims = [max(x.size(dim) for x in tens_list)
                    for dim in range(ndims)]
        padded_tensors = []
        for tens in tens_list:
            padding_tuple = tuple(x for d in range(ndims)
                                  for x in (max_dims[d] - tens.size(d), 0))
            if tens.dtype == torch.bool:
                pad_value = False
            else:
                pad_value = -1
            padded_tensors.append(
                F.pad(tens, padding_tuple[::-1], value=pad_value))
        return torch.stack(padded_tensors)

    return list(map(pad_inputs, (area_target, b2b_connectivity, p2b_connectivity,
                                     pins_pos, placement_constraints))), [pad_polygons(fp_sol), torch.stack(metrics_sol)]
    

def floorplan_collate_lite(batch):
    """Function to collate Lite (Training)"""
    area_target = [item['input'][0] for item in batch]
    b2b_connectivity = [item['input'][1] for item in batch]
    p2b_connectivity = [item['input'][2] for item in batch]
    pins_pos = [item['input'][3] for item in batch]
    placement_constraints = [item['input'][4] for item in batch]

    tree_sol = [item['label'][0] for item in batch] 
    fp_sol = [item['label'][1] for item in batch]
    metrics_sol = [item['label'][2] for item in batch]

    def pad_to_largest(tens_list):

        ndims = tens_list[0].ndim
        max_dims = [max(x.size(dim) for x in tens_list)
                    for dim in range(ndims)]
        padded_tensors = []
        for tens in tens_list:
            padding_tuple = tuple(x for d in range(ndims)
                                  for x in (max_dims[d] - tens.size(d), 0))
            if tens.dtype == torch.bool:
                pad_value = False
            else:
                pad_value = -1
            padded_tensors.append(
                F.pad(tens, padding_tuple[::-1], value=pad_value))
        return torch.stack(padded_tensors)

    return list(map(pad_to_largest, (area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints, tree_sol, fp_sol, metrics_sol)))
