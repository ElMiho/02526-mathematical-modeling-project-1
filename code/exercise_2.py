# Imports
import numpy as np
import scipy
import scipy.ndimage
import matplotlib.image as mpimg

from typing import List

###############
# Problem 2.1 #
###############
testMatrix = np.array(
    [
        [[1,2,3,4],
        [1,2,3,3],
        [1,2,3,1],
        [1,2,3,2]],
        [[1,2,3,4],
        [1,2,3,3],
        [1,2,3,1],
        [1,2,3,2]]
    ])

def calculate_Vx_LL(flattened_image_vector: np.ndarray, length: int, width: int) -> np.ndarray:
    """
    Args:
        flattened_image_vector (np.ndarray): list of images flattened to single dimension
        width (int): the image width
        length (int): the length of 'flattened_image_vector' i.e. width*height*depth

    Returns:
        The gradients in the x directions
    """
    Vx = flattened_image_vector[1:] - flattened_image_vector[:-1]
    Vx = np.append(Vx,0)
    for i in range(length):
        if (i+1) % width == 0:
            Vx[i] = 0
    return Vx

def calculate_Vy_LL(flattened_image_vector: np.ndarray, length: int, width: int, height: int) -> np.ndarray:
    """
    Args:
        flattened_image_vector (np.ndarray): list of images flattened to single dimension
        width (int): the image width
        height (int): the image height
        length (int): the length of 'flattened_image_vector' i.e. width*height*depth

    Returns:
        The gradients in the y directions
    """
    Vy = flattened_image_vector[width:] - flattened_image_vector[:-width]
    Vy = np.append(Vy, np.zeros(width))
    for i in range(length):
        if i % (width * height) >= width * (height - 1):
            Vy[i] = 0
    return Vy

def calculate_Vt_LL(flattened_image_vector: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Args:
        flattened_image_vector (np.ndarray): list of images flattened to single dimension
        width (int): the image width
        length (int): the length of 'flattened_image_vector' i.e. width*height*depth

    Returns:
        The gradients in the t directions
    """
    Vt = flattened_image_vector[width * height:] - flattened_image_vector[:-width * height]
    Vt = np.append(Vt, np.zeros(width*height))
    return Vt


###############
# Problem 2.2 #
###############

def calculate_Vx_SGF(image_vector: np.ndarray, depth: int, kernel:np.ndarray = None) -> np.ndarray:
    """
    Args:
        image_vector (np.ndarray): the vector of images/frames
        depth (int): the number of images in image_vector
        kernel (np.ndarray): the kernel to use to calculate Vx
            defaults to the Prewitt kernel

    Returns:
        Convolved image
    """
    if kernel is None:
        # Use prewitt kernel as default
        kernel = np.asarray([[[1,0,-1],
                              [1,0,-1],
                              [1,0,-1]] for _ in range(depth)])
    
    Vx = scipy.ndimage.convolve(image_vector, kernel, mode="constant", cval=0.0)

    return Vx


def calculate_Vy_SGF(image_vector: np.ndarray, depth: int, kernel:np.ndarray = None) -> np.ndarray:
    """
    Args:
        image_vector (np.ndarray): the vector of images/frames
        depth (int): the number of images in image_vector
        kernel (np.ndarray): the kernel to use to calculate Vx
            defaults to the Prewitt kernel

    Returns:
        Convolved image
    """

    if kernel is None:
        # Use prewitt kernel as default
        kernel = np.asarray([[
            [1,1,1],
            [0,0,0],
            [-1,-1,-1]] for _ in range(depth)])
    
    Vy = scipy.ndimage.convolve(image_vector, kernel, mode="constant", cval=0.0)

    return Vy

def calculate_Vt_SGF(image_vector: np.ndarray, kernel:np.ndarray = None) -> np.ndarray:
    """
    Args:
        image_vector (np.ndarray): the vector of images/frames
        kernel (np.ndarray): the kernel to use to calculate Vx
            defaults to the Prewitt kernel

    Returns:
        Convolved image
    """

    # Reshape (depth, height, width) ---> (width, height, depth)
    depth, height, width = image_vector.shape
    #image_vector = np.transpose(image_vector, (2, 1, 0))

    # Generate kernel
    if kernel is None:
        # Use prewitt kernel as default
        kernel = np.asarray([[[1,0,-1],
                              [1,0,-1],
                              [1,0,-1]] for _ in range(width)])
    
    Vt = scipy.ndimage.convolve(image_vector, kernel, mode="constant", cval=0.0)

    return Vt


###############
# Problem 2.3 #
###############

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html





















