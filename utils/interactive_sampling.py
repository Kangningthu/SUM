import os
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def add_jitter_to_bounding_boxes(bboxes: torch.Tensor, image_width: int, image_height: int,
                                 std_percentage: float = 0.1) -> torch.Tensor:
    """
    Add jitter to the bounding boxes by adding random noise to each coordinate.
    The noise is sampled from a normal distribution with mean 0 and standard deviation
    equal to a percentage of the box side length. Noise is independently sampled for each bounding box.

    Args:
        bboxes (torch.Tensor): A tensor containing bounding boxes with shape (batch_size, 1, 4).
                               Each bounding box is represented by [x_min, y_min, x_max, y_max].
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        std_percentage (float): Standard deviation as a percentage of the box side length.

    Returns:
        torch.Tensor: A tensor containing the jittered bounding boxes.
    """

    # Check if input is a PyTorch tensor
    if not torch.is_tensor(bboxes):
        raise ValueError("Input bboxes should be a PyTorch tensor")

    # Initialize tensor to store jittered bounding boxes
    jittered_bboxes = torch.zeros_like(bboxes)

    # Iterate over each bounding box in the batch
    for i in range(bboxes.size(0)):
        # Extract x_min, y_min, x_max, y_max for the current bounding box
        x_min, y_min, x_max, y_max = bboxes[i, 0]

        # Calculate side lengths for the current bounding box
        width = x_max - x_min
        height = y_max - y_min

        # Sample noise independently for x and y coordinates
        noise_x = torch.clamp(torch.randn(1, device=bboxes.device) * width * std_percentage,0,20)
        noise_y = torch.clamp(torch.randn(1, device=bboxes.device) * height * std_percentage,0, 20)

        # Add noise to the coordinates for the current bounding box
        jittered_x_min = x_min + noise_x
        jittered_y_min = y_min + noise_y
        jittered_x_max = x_max + noise_x
        jittered_y_max = y_max + noise_y

        # Clamp the coordinates to lie within the image boundary
        jittered_x_min = torch.clamp(jittered_x_min, 0, image_width)
        jittered_y_min = torch.clamp(jittered_y_min, 0, image_height)
        jittered_x_max = torch.clamp(jittered_x_max, 0, image_width)
        jittered_y_max = torch.clamp(jittered_y_max, 0, image_height)

        # Store the jittered bounding box
        jittered_bboxes[i, 0, 0] = jittered_x_min
        jittered_bboxes[i, 0, 1] = jittered_y_min
        jittered_bboxes[i, 0, 2] = jittered_x_max
        jittered_bboxes[i, 0, 3] = jittered_y_max

    return jittered_bboxes


def sample_different_position(pred_mask: torch.Tensor, gt_mask: torch.Tensor, backup_point, prob_weights=None):
    """
    Samples a position where the prediction mask and ground truth mask are different.

    Args:
        pred_mask (torch.Tensor): The prediction mask (2D tensor).
        gt_mask (torch.Tensor): The ground truth mask (2D tensor).

    Returns:
        tuple: A tuple (position, gt_value) where:
               - position is a tensor of shape (1, 2) containing the sampled position,
               - gt_value is the value in the ground truth mask at the sampled position.
               If no differing points are found, returns (None, None).
    """
    # Check if inputs are PyTorch tensors
    if not torch.is_tensor(pred_mask) or not torch.is_tensor(gt_mask):
        raise ValueError("Input masks should be PyTorch tensors")

    # Check if the masks have the same shape
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("Input masks should have the same shape")

    # Finding the indices of positions where the masks are different
    differing_points = torch.nonzero(pred_mask != gt_mask)

    # If there are no differing points, return (None, None)
    if differing_points.size(0) == 0:
        # raise ValueError("two masks are the same! ")

        return backup_point, torch.ones(1, dtype=torch.int)


    # If probability weights are provided, sample according to the weights
    if prob_weights is not None:

        # Compute flat indices from differing_points
        flat_indices = differing_points[:, 0] * pred_mask.shape[1] + differing_points[:, 1]

        # Retrieve the corresponding weights
        weights = prob_weights.flatten()[flat_indices]


        # Handle zero weights
        if weights.sum() == 0:
            return backup_point, torch.ones(1, dtype=torch.int)

        # Sample from weights
        sample_idx_in_differing_points = torch.multinomial(weights, 1)

        # Get the sampled point
        sample_idx = differing_points[sample_idx_in_differing_points].float().view(1, -1)

    else:
        index = torch.randint(0, differing_points.size(0), (1,))
        sample_idx = differing_points[index].float()

    sampled_position = sample_idx

    # Getting the value from the ground truth mask at the sampled position
    gt_value = gt_mask[tuple(sampled_position[0].long())]

    # Returning as (position tensor with the shape (1, 2), gt_value)
    return torch.flip(sampled_position, [1]), gt_value


def sample_positive_point_from_binary_mask(mask: torch.Tensor, backup_point, prob_weights=None) -> torch.Tensor:
    """
    Samples a point from a binary mask where the mask label is positive.

    Args:
        mask (torch.Tensor): A binary mask (2D tensor).

    Returns:
        torch.Tensor: A tensor of shape (1, 2) containing the sampled point, or None if no positive points.
    """
    # Check if input is a PyTorch tensor
    if not torch.is_tensor(mask):
        raise ValueError("Input mask should be a PyTorch tensor")

    # Finding the indices of positive points
    positive_points = torch.nonzero(mask == 1)

    # If there are no positive points, return backup_point
    if positive_points.size(0) == 0:
        return backup_point

    # If probability weights are provided, sample according to the weights
    if prob_weights is not None:

        flat_indices = positive_points[:, 0] * mask.shape[1] + positive_points[:, 1]

        # Retrieve the corresponding weights
        weights = prob_weights.flatten()[flat_indices]

        if weights.sum() == 0:
            print("weights sum is zero init! ")
            return backup_point

        # Sample from weights
        sample_idx_in_positive_points = torch.multinomial(weights, 1)

        # Get the sampled point
        sample_idx = positive_points[sample_idx_in_positive_points].float()
    else:
        index = torch.randint(0, positive_points.size(0), (1,))
        sample_idx = positive_points[index].float()

    sample_idx = sample_idx.view(1, -1)
    sampled_point = sample_idx

    return torch.flip(sampled_point, [1])


def sample_point_frommask(gt_binary_mask_stack, backup_points, binary_mask=None, previous_points=None, device=0, prob_weights=None):
    """
    random sample point from the ground truth mask
    Args:
        gt_binary_mask_stack: the ground truth mask
        binary_mask: the predicted mask
        previous_points: the previous sampled points
    """
    coords_torch_list = []

    if binary_mask is None or previous_points is None:
        input_label_torch = torch.ones((gt_binary_mask_stack.size()[0], 1), dtype=torch.int)
    else:
        input_label_torch = torch.ones((previous_points[1].size()[0], previous_points[1].size()[1] + 1),
                                       dtype=torch.int)
        input_label_torch[:, :-1] = previous_points[1]

    for mask_index in range(gt_binary_mask_stack.size()[0]):
        gt_mask = gt_binary_mask_stack[mask_index][0]  # 256, 256

        backup_point = backup_points[mask_index]
        assert len(gt_mask.size()) == 2  # two dimension case

        if prob_weights is not None:
            mask_prob_weights = prob_weights[mask_index][0]  # 256, 256
        else:
            mask_prob_weights = None

        if not binary_mask is None:
            # assert previous_points is not None

            pred_mask = binary_mask[mask_index][0]  # 256, 256
            sampled_position, gt_value = sample_different_position(pred_mask, gt_mask, backup_point,  prob_weights=mask_prob_weights)
            # sampled_position size is (1, 2)
            # previous_points is tuple, previous_points[0] is the coords_torch_all, get the corresponding coords_torch
            if previous_points is not None:
                coords_torch = torch.cat((previous_points[0][mask_index], sampled_position.to(device)), dim=0)
                coords_torch_list.append(coords_torch)
                input_label_torch[mask_index, -1] = gt_value  # assign the label for that point
            else:
                coords_torch = sampled_position
                coords_torch_list.append(coords_torch)
                input_label_torch[mask_index, -1] = gt_value
        else:
            coords_torch = sample_positive_point_from_binary_mask(gt_mask, backup_point,  prob_weights=mask_prob_weights)
            coords_torch_list.append(coords_torch)

    coords_torch_all = torch.stack(coords_torch_list, dim=0)
    return (coords_torch_all.to(device), input_label_torch.to(device))
