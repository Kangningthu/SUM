import torch
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
from PIL import Image
import pandas as pd

from utils.transforms import ResizeLongestSide
from torch.nn import functional as F
import random
import os
import tarfile
import numpy as np
from io import BytesIO
from PIL import Image
# from utils.amg import batched_mask_to_box
import tempfile
from PIL import Image

def select_and_sort(input_list, target_number):
    """
    If the size of the input_list is larger than the target_number,
    randomly select target_number of elements from input_list and
    return them in sorted order.

    Args:
        input_list (list): The input list.
        target_number (int): The target number of elements.

    Returns:
        list: The sorted list with target_number of elements.
    """
    # If the size of the input_list is larger than the target_number,
    # select a subset of elements and sort them.
    if len(input_list) > target_number:
        # Randomly select elements
        selected_elements = random.sample(input_list, target_number)
        # Sort the selected elements
        return sorted(selected_elements)
    else:
        # If size is less than or equal to target_number, return the input_list as is
        return input_list


def pad_to_batch_size(source, target_batch_size):
    """
    Pads the source tensor with zeros to match the target batch size.

    Args:
        source (torch.Tensor): The source tensor. Shape (batch_size, *).
        target_batch_size (int): The target batch size.

    Returns:
        torch.Tensor: The padded tensor with shape (target_batch_size, *).
    """
    # Get the dimensions excluding the batch dimension
    data_dims = source.size()[1:]

    # Create a tensor of zeros with the target size
    padded_tensor = torch.zeros((target_batch_size, *data_dims), dtype=source.dtype, device=source.device)

    # Assign the elements from the source tensor to the padded tensor
    padded_tensor[:source.size(0)] = source

    return padded_tensor


def filter_table(df, min_ratio, max_ratio):
    """filter the tables to select the mask within the specific range"""
    # Convert the 'crop_box_w', 'crop_box_h', and 'area' to integers
    df['crop_box_w'] = df['crop_box_w'].astype(int)
    df['crop_box_h'] = df['crop_box_h'].astype(int)
    df['area'] = df['area'].astype(int)


    # Find the maximum dimension among 'crop_box_w' and 'crop_box_h'
    max_box_dimension = df[['crop_box_w', 'crop_box_h']].max(axis=1)

    # Compute the ratio and store it in a new column
    df['ratio'] = df['area'] / (df['crop_box_w'] * df['crop_box_h'])
    df['resize_area'] = df['area'] * ((1024/max_box_dimension)**2)

    # Use boolean indexing to filter the DataFrame
    filtered_df = df[(df['resize_area'] >= 100*min_ratio) & (df['ratio'] <= max_ratio)]

    # Return the filtered DataFrame
    return filtered_df





def load_directory_list(file_path):
    try:
        # Open and read the file
        with open(file_path, 'r') as file:
            # Read all lines into a list
            directory_list = file.readlines()

        # Remove any newlines or whitespaces from each element
        directory_list = [line.strip() for line in directory_list]

        return directory_list

    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def read_files_in_tar(tar_path):
    """
    Read the contents of files within a tar archive into memory and process them
    based on file extension.

    :param tar_path: The file path to the tar archive.
    """
    # Open the tar file
    dict_tar = {}
    with tarfile.open(tar_path, 'r') as archive:
        # Iterate through the items in the archive
        for item in archive:
            # Check if it is a file
            if item.isfile():
                # Extract the file object
                file = archive.extractfile(item)
                # Read the contents of the file
                contents = file.read()

                # Process the contents based on file extension
                if item.name.endswith('.png'):
                    # Convert bytes to an image
                    image = Image.open(BytesIO(contents))
                    # Convert image to numpy array
                    image_array = np.array(image)
                    # print(f"Contents of {item.name} as numpy array:")
                    # print(image_array)
                    dict_tar[item.name] = image_array

                elif item.name.endswith('.csv'):
                    # Convert bytes to a pandas DataFrame
                    df = pd.read_csv(BytesIO(contents))
                    # print(f"Contents of {item.name} as pandas DataFrame:")
                    # print(df)
                    dict_tar[item.name] = df

    return dict_tar


def load_proba_from_tar_png(tar_file_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        with tarfile.open(tar_file_path, 'r') as tar:
            tar.extractall(path=tmpdirname)
            images_dict = {}
            for filename in sorted(os.listdir(tmpdirname), key=lambda x: int(x.split('.')[0])):
                # Load the PNG image
                img_path = os.path.join(tmpdirname, filename)
                img = Image.open(img_path)

                # Convert the image to an np.array and rescale to the 0-1 range
                arr = np.array(img).astype(np.float32) / 255.0

                # Reshape to (1, h, w)
                reshaped_arr = arr[np.newaxis, np.newaxis, ...]

                # Insert into the dictionary with filename as the key
                images_dict[filename] = reshaped_arr

    return images_dict



def inverse_sigmoid(y):
    y = np.clip(y,  7e-5, 0.9995)  # Clipping to avoid log(0)
    return np.log(y / (1 - y))
#

def convert_format_input(pred_logit,rescaled_logit=False,return_prob=False):
    pseudo_label_converted = (pred_logit/255).astype(np.float16)
    if return_prob:
        return pseudo_label_converted
    # probabilities = expit(pseudo_label_converted)  # Apply sigmoid to get probabilities
    inverse_probabilities = inverse_sigmoid(pseudo_label_converted)  # Apply inverse sigmoid
    if rescaled_logit:
        inverse_probabilities = inverse_probabilities*10 # compensate for the rescaling of the logit
    inverse_probabilities = np.clip(inverse_probabilities, -10e5,10e5)
    return inverse_probabilities  # Return as numpy array





class SAM1BDataset(Dataset):
    def __init__(self, num_images=10000, img_size=1024, datalistdir = 'sam1b_pseudolabel_0_115000.txt', min_ratio=1,max_ratio=0.95, max_num_mask=64,return_binary=True,return_all_mask=False, return_prob=False, start_idx=0, image_root='/mnt/localssd/SAM1B'):
        """
        Args:
            data_list (list): List of your data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.qfile = load_directory_list(datalistdir)[start_idx:start_idx+num_images]
        # self.data_list = data_list
        self.transform = ResizeLongestSide(1024)
        # self.sam_model =sam_model
        self.num_images = num_images
        pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        pixel_std = [58.395, 57.12, 57.375]
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        self.img_size = img_size
        # self.maskrootdir = maskrootdir
        #  maskrootdir='/home/kangningl/data/s3'
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.max_num_mask = max_num_mask


        print('load probs rather than binary labels')
        self.load_prob = True
        self.return_binary = return_binary
        self.return_prob = return_prob
            # if both false, return the logit

        self.return_all_mask = return_all_mask # to return all the possible mask after the filtering, so that

        print(f"load {len(self.qfile)} images")
        print(f"max {num_images} images")
        self.num_images = min(num_images, len(self.qfile))
        self.image_root = image_root


    def __len__(self):
        # return self.num_images
        len_dataset = min(self.num_images, len(self.qfile))
        print(f"Dataset length reported as {len_dataset}")
        return len_dataset
        # return len(self.data_list)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def pad_label(self, x: torch.Tensor) -> torch.Tensor:
        """pad label to batchfy."""
        # Normalize colors
        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        # Get the data sample at index idx
        # sample = self.data_list[idx]
        index_str = self.qfile[idx]
        file_path = index_str   # /mnt/localssd/sam_pseudolabels/sa_5122.jpg.tar




        if not os.path.exists(file_path):
            idx_next = random.randint(0, self.num_images-1)
            sample = self.__getitem__(idx_next)
            return sample

        tempt_list = self.qfile[idx].split('/')
        image_name = tempt_list[-1].split('.')[0] 

        try:
            image = np.array(Image.open(os.path.join(self.image_root,image_name+'.jpg')))
        except Exception:
            idx_next = random.randint(0, self.num_images-1)
            sample = self.__getitem__(idx_next)
            return sample

        # if the image has the alpha map, or the image size is too large
        if image.shape[2] != 3 or image.shape[0] > 5000 or image.shape[1] > 5000:
            # sample = self.__getitem__(idx + 1)
            print(f"Failed to load {index_str}th image due to the alpha map or the image size is too large")
            idx_next = random.randint(0, self.num_images - 1)
            sample = self.__getitem__(idx_next)
            return sample




        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        # if the image has the alpha map, or the image size is too large
        if transformed_image.size()[1] != 3 or transformed_image.size()[2] > 5000 or transformed_image.size()[3] > 5000:
            # sample = self.__getitem__(idx + 1)
            idx_next = random.randint(0, self.num_images-1)
            sample = self.__getitem__(idx_next)
            return sample

        input_image = self.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])

        dict_mask = read_files_in_tar(file_path)
        mask_meta_data = dict_mask['metadata.csv']

        mask_meta_data = filter_table(mask_meta_data, min_ratio=self.min_ratio, max_ratio=self.max_ratio)

        num_masks_total = len(mask_meta_data)

        filtered_index = (mask_meta_data.index).tolist()



        if num_masks_total == 0:
            # skip this sample
            # get to the next one
            idx_next = random.randint(0, self.num_images-1)
            sample = self.__getitem__(idx_next)
            return sample

        coords_torch_list = []
        box_torch_list = []
        gt_binary_mask_list = []

        if self.return_all_mask:
            self.max_num_mask = num_masks_total


        input_label_torch = torch.ones((self.max_num_mask, 1), dtype=torch.int)

        if num_masks_total > self.max_num_mask:

            candidates_list = select_and_sort(filtered_index, self.max_num_mask)
            # print(candidates_list)
        else:
            candidates_list = filtered_index


        for mask_index in candidates_list:
            input_point_ori = np.array(
                [[mask_meta_data['point_input_x'][mask_index], mask_meta_data['point_input_y'][mask_index]]])

            input_point = self.transform.apply_coords(input_point_ori, original_image_size)
            coords_torch = torch.as_tensor(input_point, dtype=torch.float)
            coords_torch_list.append(coords_torch)

            # the box coordinates are in the format of [x0, y0, x1, y1]
            box3 = mask_meta_data['bbox_x0'][mask_index] + mask_meta_data['bbox_w'][mask_index]
            box4 = mask_meta_data['bbox_y0'][mask_index] + mask_meta_data['bbox_h'][mask_index]
            prompt_box = np.array(
                [[mask_meta_data['bbox_x0'][mask_index], mask_meta_data['bbox_y0'][mask_index], box3, box4]])
            box = self.transform.apply_boxes(prompt_box, original_image_size)
            box_torch = torch.as_tensor(box, dtype=torch.float) #
            box_torch_list.append(box_torch)

            if self.load_prob:
                gt_grayscale = dict_mask[str(mask_index).zfill(3) + '_probs.png']

                # convert the format
                gt_grayscale = convert_format_input(gt_grayscale, rescaled_logit=False, return_prob=self.return_prob)

                gt_mask_resized = torch.from_numpy(
                    np.resize(gt_grayscale, (1, 1, gt_grayscale.shape[0], gt_grayscale.shape[1])))
                if self.return_binary:
                    # binarize the logit
                    if self.return_prob:
                        gt_mask_resized = gt_mask_resized>0.5
                    else:
                        gt_mask_resized = gt_mask_resized>0

                gt_binary_mask = torch.as_tensor(gt_mask_resized, dtype=torch.float32)

            else:
                gt_grayscale = dict_mask[str(mask_index).zfill(3) + '.png']

                gt_mask_resized = torch.from_numpy(
                    np.resize(gt_grayscale, (1, 1, gt_grayscale.shape[0], gt_grayscale.shape[1])))
                gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
            gt_binary_mask_list.append(gt_binary_mask)

        coords_torch_all = torch.stack(coords_torch_list, dim=0)

        box_torch_stack = torch.stack(box_torch_list, dim=0)  # 64, 1, 4
        gt_binary_mask_stack = torch.cat(gt_binary_mask_list, dim=0)  # 64, 1, 1024, 1024
        if num_masks_total < self.max_num_mask:
            box_torch_stack = pad_to_batch_size(box_torch_stack, self.max_num_mask)
            gt_binary_mask_stack = pad_to_batch_size(gt_binary_mask_stack, self.max_num_mask)
            coords_torch_all = pad_to_batch_size(coords_torch_all, self.max_num_mask)

        # points = (coords_torch_all, input_label_torch)  # (64, 1, 2) (64)
        gt_binary_mask_stack = F.interpolate(gt_binary_mask_stack, input_size, mode="nearest")
        gt_binary_mask_stack = self.pad_label(gt_binary_mask_stack)



        sample = {'image': input_image, 'boxes': box_torch_stack, 'shape': torch.tensor([input_size[0],input_size[1]]),
                  'label': gt_binary_mask_stack, 'original_image_size': torch.tensor([original_image_size[0],original_image_size[1]]),
                  'num_masks_total': num_masks_total, 'imidx': idx, 'candidates_list': torch.tensor(np.array(candidates_list)), 'raw_name': image_name}

        return sample


def postprocess_masks_ori_res(masks):
    masks = torch.Tensor(masks)
    masks = F.interpolate(masks, (1024, 1024), mode="bilinear", align_corners=False)
    # masks = masks[..., :input_size[0], :input_size[1]]
    return masks



class SAM1BUncerAwareDataset(SAM1BDataset):
    def __init__(self, num_images=10000, img_size=1024,
                 datalistdir='sam_refine_label_png_files_list.txt',
                 min_ratio=1, max_ratio=0.95, max_num_mask=64, return_final_binary=False, return_all_mask=False, min_refine_ratio=10, use_uncertainmap=True, start_idx=0, uncertain_map_folder_path='xxx'):
        super().__init__(num_images=num_images, img_size=img_size, datalistdir=datalistdir, min_ratio=min_ratio,
                         max_ratio=max_ratio, max_num_mask=max_num_mask, return_binary=False,
                         return_all_mask=return_all_mask, return_prob=True, start_idx=start_idx)


        self.uncertain_map_folder_path = uncertain_map_folder_path
        self.min_refine_ratio = min_refine_ratio
        self.return_final_binary = return_final_binary
        self.use_uncertainmap = use_uncertainmap

        # self.min_ratio = min_ratio
        # self.max_ratio = max_ratio
        print('min_ratio'+ str(self.min_ratio))
        print('max_ratio' + str(self.max_ratio))
        print('min_refine_ratio' + str(self.min_refine_ratio))




    def __getitem__(self, idx):

        sample = super().__getitem__(idx)
        if not self.use_uncertainmap:
            if self.return_final_binary:
                sample['label'] = (sample['label'] > 0.5).to(torch.float32)
            return sample

        candidates_list = sample['candidates_list'].numpy()
        index_str = self.qfile[idx] 

        file_path = index_str  

        tempt_list = self.qfile[idx].split('/')
        file_name = tempt_list[-1].split('.')[0]


        dict_mask = read_files_in_tar(file_path) # the path for the pseudo label of SAM
        mask_meta_data = dict_mask['metadata.csv']

        mask_meta_data = filter_table(mask_meta_data, min_ratio=self.min_refine_ratio, max_ratio=self.max_ratio)

        filtered_index_refined = (mask_meta_data.index).tolist()
        try:
            dict_tar_pseudo = load_proba_from_tar_png(os.path.join( self.uncertain_map_folder_path,file_name+'.tar'))

            uncertain_map_list = []
            for indx_count, mask_index in enumerate(candidates_list):
                mask_name = str(mask_index) + '.png'
                # only get the refinement pseudo label when we have it and the mask is filtered to be within the range
                if mask_name in dict_tar_pseudo.keys() and mask_index in filtered_index_refined:

                    low_res_refine = dict_tar_pseudo[mask_name]

                    hr_res_refine = postprocess_masks_ori_res(low_res_refine)  # 1,1,1024,1024

                    binary_prob = sample['label'][indx_count]

                    uncertain_map_list.append(torch.abs(binary_prob - hr_res_refine))
                else:
                    uncertain_map_list.append(
                        torch.zeros((1, 1, 1024, 1024)))  # the uncertain map is all zero as no map found

            uncertain_map_stack = torch.cat(uncertain_map_list, dim=0)  # 64,1,1024,1024
        except:
            print('no refined label found for this image, sample next' + file_name)
            idx_next = random.randint(0, self.num_images - 1)
            sample = self.__getitem__(idx_next)
            return sample

            # uncertain_map_stack = torch.zeros((self.max_num_mask, 1, 1024, 1024))  # 64,1,1024,1024


        sample['uncertain_map_stack'] = uncertain_map_stack
        if self.return_final_binary:
            sample['label'] = (sample['label']>0.5).to(torch.float32)

        return sample



def random_sample(arr, size):
    """
    Randomly sample elements from an array.
    """
    if arr.shape[0] >= size:
        indices = np.random.choice(arr.shape[0], size, replace=False)
        return arr[indices]
    else:
        return arr
