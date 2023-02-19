# Imports #
import numpy as np
import scipy
import scipy.ndimage
import matplotlib as plt
import PIL
import matplotlib.pyplot as plt
from typing import List

#########################
# Gradient calculations #
#########################

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

def calculate_Vx_SGF(image_vector: np.ndarray, kernel:np.ndarray = None, mode:str = "nearest") -> np.ndarray:
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
        kernel = np.flip([[1,0,-1],
                          [1,0,-1],
                          [1,0,-1]])
    
    Vx = np.asarray([scipy.ndimage.convolve(image, kernel, mode=mode) for image in image_vector])

    return Vx

def calculate_Vy_SGF(image_vector: np.ndarray, kernel:np.ndarray = None, mode:str = "nearest") -> np.ndarray:
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
        kernel = np.flip([[1,1,1],[0,0,0],[-1,-1,-1]])
    
    Vy = np.asarray([scipy.ndimage.convolve(image, kernel, mode=mode) for image in image_vector])

    return Vy

def calculate_Vt_SGF(image_vector: np.ndarray, kernel:np.ndarray = None, mode:str = "nearest") -> np.ndarray:



    """
    Args:
        image_vector (np.ndarray): the vector of images/frames
        kernel (np.ndarray): the kernel to use to calculate Vx
            defaults to the Prewitt kernel
    Returns:
        Convolved image
    """

    # Reshape (depth, height, width) ---> (width, height, depth)
    image_vector = np.transpose(image_vector, (2,1,0))

    # Generate kernel
    if kernel is None:
        # Use prewitt kernel as default
        kernel = np.flip([[1,0,-1],
                          [1,0,-1],
                          [1,0,-1]])
    
    # Reshape (width, height, depth) ---> input shape
    Vt = np.asarray([scipy.ndimage.convolve(image, kernel, mode=mode) for image in image_vector])
    Vt = np.transpose(Vt, (2,1,0))

    return Vt

######################
# Solve optical flow #
######################

def return_A_b(vx_N, vy_N, vt_N):
    vx_N = np.reshape(vx_N, (-1, 1))
    vy_N = np.reshape(vy_N, (-1, 1))
    b = -np.reshape(vt_N, (-1, 1))
    A = np.hstack((vx_N, vy_N))
    return A, b

def solve_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Args:
        A (np.ndarray): the system matrix
        b (np.ndarray): the right hand side
    Returns
        The least squares solution to Ax = b
    """
    return np.linalg.lstsq(A, b,rcond=None)[0]

def optical_flow(images: np.ndarray, interval: int, N: int):
    """
    Parameters
    ----------
    images : 3D-array containing movie
    interval : integer stating how many optical flow vectors to calculate
    N : Integer determing how big a neighboorhood to use in Lucas-Kanaade method
    Returns
    -------
    x_sol : 3D-array with x-coordinates for opticalflow vectors
    y_sol : 3D-array with y-coordinates for opticalflow vectors
    """
    depth, height, width = images.shape
    # ensure N is odd.
    if N%2 == 0:
        N +=1
    
    Vt = calculate_Vt_SGF(images) #(64,255,255)
    Vx = calculate_Vx_SGF(images) #(64,255,255)
    Vy = calculate_Vy_SGF(images) #(64,255,255)

    # flattened_images = np.reshape(images, (-1,))
    # Vt = np.reshape(calculate_Vt_LL(flattened_images, width, height), (depth, height, width)) #(64,255,255)
    # Vx = np.reshape(calculate_Vx_LL(flattened_images, depth, width), (depth, height, width)) #(64,255,255)
    # Vy = np.reshape(calculate_Vy_LL(flattened_images, depth, width, height), (depth, height, width)) #(64,255,255)
    
    
    x_sol = np.zeros((depth,height,width)) 
    y_sol = np.zeros((depth,height,width))
    for frame in range(depth-1):
        for j in range(N//2, height, interval):
            for k in range(N//2, width, interval):
                V_x_N = Vx[frame,j-N//2: j+N//2+1, k-N//2: k+N//2+1]
                V_y_N = Vy[frame,j-N//2: j+N//2+1, k-N//2: k+N//2+1]
                V_t_N = Vt[frame,j-N//2: j+N//2+1, k-N//2: k+N//2+1]
                if any(np.abs(np.reshape(V_x_N,(-1,)))>0.5) or any(np.abs(np.reshape(V_y_N,(-1,)))>0.5):
                    A,b = return_A_b(V_x_N, V_y_N, V_t_N)
                    opticFlow = solve_least_squares(A, b)
                else:
                    opticFlow = [0,0]
                x_sol[frame, j, k] = opticFlow[0]
                y_sol[frame, j, k] = opticFlow[1]
    return x_sol, y_sol

#####################
# General functions #
#####################

def plotVectorField(frame: np.ndarray, opticFlowX: np.ndarray, opticFlowY: np.ndarray, frameNr: int):
    """
    Input: frame: An nxm image
                opticFlowX = x-coordinates of optical flow vectors
                opticFlowY = y-coordinates of optical flow vectors
    Return: Save an image of vectorfield plotted on image     
    """
    #Compute coordiantes for vectors
    height, width = frame.shape
    idx_x = np.arange(width)
    idx_y = np.arange(height)
    idx_x,idx_y = np.meshgrid(idx_x, idx_y)
    
    #Ignore all nonzero entries
    mask1 = np.logical_and(np.abs(opticFlowX) > 0.1, np.abs(opticFlowY) > 0.1)
    mask2 = np.logical_and(np.abs(opticFlowX) < 10, np.abs(opticFlowY) < 10)
    mask = np.logical_and(mask1,mask2)

    X = idx_x[mask]
    Y = idx_y[mask]
    U = opticFlowX[mask]
    V = opticFlowY[mask]

    
    #Plot
    fig, ax = plt.subplots()
    ax.imshow(frame,cmap="gray")
    ax.quiver(X,Y,U,V, scale = 100, width=0.01, color='b')
    plt.savefig(f"toyProblem_F22_vectorField/{frameNr}.jpg")

def importImages():
    """
    Returns a 3D-array of images
    """
    images = []
    for i in range(1, 65):
        num = str(i) if i >= 10 else "0" + str(i)
        image = PIL.Image.open(f"./toyProblem_F22/frame_{num}.png").convert("L")
        images.append(np.asarray(image, dtype=np.float32)/255)

    # Convert list of images to np format
    images = np.asarray(images)
    return images


