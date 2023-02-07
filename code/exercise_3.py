import numpy as np

from typing import List

###############
# Problem 3.1 #
###############

def setupEquation2(N: int, height:int, width:int, frame: int,
                    Vx: np.ndarray, Vy: np.ndarray, Vt: np.ndarray, 
                    offset_x: int, offset_y: int) -> List[np.ndarray]:
    """
    Args: 
        N (int): the size of the cutout
        height (int): the height of the image
        width (int): the width of the image
        frame (int): the frame in the video 
        Vx (np.ndarray): the x-gradients
        Vy (np.ndarray): the y-gradients
        Vt (np.ndarray): the time-gradients
        offset_x (int): the starting x of the cutout
        offset_y (int): the starting y of the cutout

    Returns:
        The matrix A and vector b in equation (2)
    """
    # Generate subsets of Vx, Vy, Vz
    V_x_N = Vx[frame,offset_y: min(offset_y+N,height), offset_x: min(offset_x+N,width)]
    V_y_N = Vy[frame,offset_y: min(offset_y+N,height), offset_x: min(offset_x+N,width)]
    V_t_N = Vt[frame,offset_y: min(offset_y+N,height), offset_x: min(offset_x+N,width)]

    # Generate A and b
    A = np.concatenate([np.reshape(V_x_N, (-1,)), np.reshape(V_y_N, -1)])
    b = np.reshape(V_t_N, (-1,))

    return A, b


def solveLeastSquares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Args:
        A (np.ndarray): the system matrix
        b (np.ndarray): the right hand side
    
    Returns
        The least squares solution to Ax = b
    """

    return np.linalg.lstsq(A, b)








