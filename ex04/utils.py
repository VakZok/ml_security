import datetime
import requests
import socket
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

setup = None
dm = None
ds = None
imagenet_mean = [0.485, 0.456, 0.406]
cifar10_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
imagenet_std = [0.229, 0.224, 0.225]
cifar10_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]

def system_startup():
    """
    This function is called at the beginning of the notebook to setup the system.
    It initializes global variables, logs system information and returns a dictionary with the setup information.

    return:
        setup: dictionary with device and dtype
    """
    # Choose GPU device and print status information:
    global setup, dm, ds
    __load_imagenet_class_names()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float)  # non_blocking=NON_BLOCKING
    print('Currently evaluating -------------------------------:')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')
    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')
    else:
        print('No GPU available, using the CPU instead.')

    dm = torch.as_tensor(imagenet_mean, **setup)[:, None, None]
    ds = torch.as_tensor(imagenet_std, **setup)[:, None, None]
    return setup

def load_img(path: str, size=None):
    """
    Loads an image from the specified file path, optionally resizing it, and converts it to a normalized tensor.
    Args:
        path (str): The file path to the image.
        size (tuple, optional): A tuple specifying the desired size (width, height) to resize the image to. 
                                If None, the image is not resized.
    Returns:
        torch.Tensor: A tensor representation of the image, normalized and permuted to have channels first.
    """
    img = Image.open(path)
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    car_np = np.array(img)/255
    gt_img = torch.as_tensor(car_np, **setup).permute(2,0,1)
    gt_img = normalize(gt_img)
    return gt_img

def plot(tensor):
    """
    Plots a tensor as an image or a grid of images using matplotlib.
    Parameters:
        tensor (torch.Tensor): A 3D or 4D tensor to be plotted. If the tensor is 3D, it is assumed to be a single image.
                           If the tensor is 4D, it is assumed to be a batch of images.
    Returns:
        None: Displays the image or grid of images using matplotlib's imshow function.
    Notes:
        - The tensor is expected to be in the format (C, H, W) for a single image or (N, C, H, W) for a batch of images,
          where C is the number of channels, H is the height, and W is the width.
    """
    tensor = tensor.clone().detach()
    if len(tensor.shape) == 3:
        return plt.imshow(tensor.permute(1, 2, 0).cpu());
    elif tensor.shape[0] == 1:
        return plt.imshow(tensor[0].permute(1, 2, 0).cpu());
    else:
        n = tensor.shape[0]
        grid_size = min(tensor.shape[0], max(4, int(np.ceil(np.sqrt(n)))))
        cols = grid_size
        rows = int(np.ceil((tensor.shape[0])/grid_size))
        fig, axes = plt.subplots(rows, cols, figsize=(min(3*cols, 12), min(3*rows, 12)))
        for i, im in enumerate(tensor):
            row, col = divmod(i, grid_size)
            if rows == 1:
                axes[col].imshow(im.permute(1, 2, 0).cpu())
                axes[col].axis('off')
            else:
                axes[row, col].imshow(im.permute(1, 2, 0).cpu())
                axes[row, col].axis('off')
        # Hide any unused subplots
        for j in range(i + 1, grid_size * rows):
            row, col = divmod(j, grid_size)
            if rows == 1:
                axes[col].axis('off')
            else:
                axes[row, col].axis('off')

def denormalize(tensor):
    """
    Denormalizes a given tensor.
    This function takes a tensor and applies a denormalization transformation
    using predefined scaling (ds) and mean (dm) values. The resulting tensor
    values are clamped between 0 and 1.

    Parameters:
        tensor (torch.Tensor): The input tensor to be denormalized.
    Returns:
        torch.Tensor: The denormalized tensor with values clamped between 0 and 1.
    Note:
        This function may be used to denormalize pictures, e.g., for display purposes.
    """
    return (tensor*ds+dm).clamp_(0, 1)

def normalize(tensor):
    """
    Normalize a given tensor using ImageNet mean (dm) and standard deviation (ds).
    The retruned tensor may be used as input to a pre-trained model.

    Parameters:
        tensor (torch.Tensor): The input tensor to be normalized.
    Returns:
        torch.Tensor: The normalized tensor.
    """

    return (tensor - dm)/ds

def __load_imagenet_class_names():
    global classes
    # URL for the ImageNet class index file
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

    # Download the class names
    response = requests.get(url)
    classes = response.json()