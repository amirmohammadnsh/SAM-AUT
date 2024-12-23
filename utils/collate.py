import torch
import numpy as np

def pad_tensors_to_max_rows(tensors):
    """
    Pad tensors along the first dimension to match the maximum number of rows.
    Args:
        tensors (list): A list of PyTorch tensors.
    Returns:
        list: A list of padded tensors.
    """
    all2d = True
    for i, tensor in enumerate(tensors):
        if len(tensor.shape) == 2 and all2d and tensor.shape[1]==4:  # Check if the tensor has 2 dimensions
            continue  
        elif len(tensor.shape) != 2:
            all2d = False
        elif len(tensor.shape) == 2 and not all2d:
            tensors[i] = tensor.unsqueeze(0)
        elif len(tensor.shape) == 2 and all2d and tensor.shape[1]==256:
            tensors[i] = tensor.unsqueeze(0)
            all2d = False

    padded_tensors = []
    if all2d:
        max_rows = max(tensor.size(0) for tensor in tensors)
        
        for tensor in tensors:
            padding_rows = max_rows - tensor.size(0)
            padding = (0, 0, 0, padding_rows)  # Pad along the first dimension only
            padded_tensor = torch.nn.functional.pad(tensor, padding)
            padded_tensors.append(padded_tensor)
        return padded_tensors
    else:
        max_dim_0 = max(tensor.shape[0] for tensor in tensors)
        for tensor in tensors:
            pad_dims = (0, 0, 0, 0, 0, max_dim_0 - tensor.shape[0])  # Pad along the first dimension
            padded_tensor = torch.nn.functional.pad(tensor, pad_dims)
            padded_tensors.append(padded_tensor)
        return padded_tensors
def pad_arrays_to_max_rows(arrays):
    """
    Pad arrays along the first axis to match the maximum number of rows.
    Args:
        arrays (list): A list of NumPy arrays.
    Returns:
        list: A list of padded arrays.
    """
    max_rows = max(array.shape[0] for array in arrays)
    padded_arrays = []
    for array in arrays:
        padding_rows = max_rows - array.shape[0]
        padding = ((0, padding_rows), (0, 0), (0, 0))  # Pad along the first axis only
        padded_array = np.pad(array, padding)
        padded_arrays.append(padded_array)
    return padded_arrays
def collate_fn(batch):
    """
    Custom collate function to handle variable size tensors in the batch.
    Args:
        batch (list): A list of samples, where each sample is a dictionary containing 'image', 'bbox', and 'mask'.
    Returns:
        dict: A dictionary containing the batched data.
    """
    pixel_values = [sample['pixel_values'] for sample in batch]
    original_sizes = [sample['original_sizes'] for sample in batch]
    input_boxes = [sample['input_boxes'] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    image_ids = [sample["image_ids"] for sample in batch]
    gt_bboxes = [sample["gt_bboxes"] for sample in batch]

    pixel_values = torch.stack(pixel_values)
    original_sizes = torch.stack(original_sizes)
    input_boxes = pad_tensors_to_max_rows(input_boxes)
    input_boxes = torch.stack(input_boxes)
    image_ids = torch.tensor(image_ids)
    labels = pad_tensors_to_max_rows(labels)
    labels = torch.stack(labels)
    gt_bboxes = pad_tensors_to_max_rows(gt_bboxes)  
    gt_bboxes = torch.stack(gt_bboxes)  

    
    # batch = {}
    batch = {
        'pixel_values': pixel_values,
        'original_sizes':original_sizes,
        'labels': labels,
        'input_boxes': input_boxes,
        'image_ids': image_ids,
        'gt_bboxes': gt_bboxes, 

    }

    return batch
