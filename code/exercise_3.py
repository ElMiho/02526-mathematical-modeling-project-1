import numpy as np
from exercise_2 import *
import PIL
from typing import List
import matplotlib.pyplot as plt

###############
# Problem 3.1 #
###############

def return_A_b(vx_N, vt_N, vy_N):
    vx_N = np.reshape(vx_N, (-1, 1))
    vy_N = np.reshape(vy_N, (-1, 1))
    vt_N = np.reshape(vt_N, (-1, 1))

    A = np.hstack((vx_N, vy_N))
    return A, vt_N


def setup_equation_2(N: int, height:int, width:int, frame: int,
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

    V_x_N = np.reshape(V_x_N, (-1, 1))
    V_y_N = np.reshape(V_t_N, (-1, 1))
    V_t_N = np.reshape(V_t_N, (-1, 1))

    A = np.hstack((V_x_N, V_y_N))
    b = V_t_N

    return A, b


def solve_least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Args:
        A (np.ndarray): the system matrix
        b (np.ndarray): the right hand side
    
    Returns
        The least squares solution to Ax = b
    """
    m, c = np.linalg.lstsq(A, b)[0]
    return m, c

def loop_image(images: np.ndarray, interval: int, N: int):
    depth, height, width = images.shape
    # ensure N is odd.
    if N%2 == 0:
        N +=1
    
    Vt = calculate_Vt_SGF(images) #(64,255,255)
    Vx = calculate_Vx_SGF(images, depth) #(64,255,255)
    Vy = calculate_Vy_SGF(images, depth) #(64,255,255)

    print("shape vt" + str(Vt.shape))
    print("shape vx" + str(Vx.shape))
    print("shape vy" + str(Vy.shape))

    x_sol = np.zeros((depth,height,width)) 
    y_sol = np.zeros((depth,height,width))

    for frame in range(depth-1):
        for j in range(N//2, width, interval):
            for k in range(N//2, height, interval):
                #setup_equation_2(N, height, width, frame, Vx, Vy, Vt, x_offset, y_offset)
                V_x_N = Vx[frame,k-N//2: k+N//2+1, j-N//2: j+N//2+1]
                V_y_N = Vy[frame,k-N//2: k+N//2+1, j-N//2: j+N//2+1]
                V_t_N = Vt[frame,k-N//2: k+N//2+1, j-N//2: j+N//2+1]
                print("vxN")
                print(V_x_N.shape)
                A,b = return_A_b(V_x_N, V_t_N, V_y_N)
                print("A")
                print(A.shape)
                print("b")
                print(b.shape)
                print("\n\n")
                res1, res2 = solve_least_squares(A, b)
                x_sol[frame, j, k] = res1[0]
                y_sol[frame, j, k] = res2[0]
                
    return x_sol, y_sol


# Save images to array
images = []
for i in range(1, 4):
    num = str(i) if i >= 10 else "0" + str(i)
    image = PIL.Image.open(f"./toyProblem_F22/frame_{num}.png").convert("L")
    images.append(np.asarray(image, dtype=np.float32)/255)

# Convert list of images to np format
images = np.asarray(images)
depth, height, width = images.shape

# Vx = calculate_Vx_SGF(images, depth)
# Vy = calculate_Vy_SGF(images, depth)
# Vt = calculate_Vt_SGF(images)

# N = 5
# x_offset = 40
# y_offset = 40
# frame = 0
# A, b = setup_equation_2(N, height, width, frame, Vx, Vy, Vt, x_offset, y_offset)

# print(A.shape)
# print(b.shape)
# sol1, sol2 = solve_least_squares(A, b)
# print("\n\nsol = " + str(sol1) + "," + str(sol2))

# fig, ax = plt.subplots()
# the_image = ax.imshow(
#     images[frame], cmap="gray"
# )
# # X, Y = np.mgrid[]
# plt.show()

# Problem 3.2
x, y = loop_image(images, 50, 3)
print(x.shape)
print(y.shape)
print(x[2,:,:])
print(y[2,:,:])
