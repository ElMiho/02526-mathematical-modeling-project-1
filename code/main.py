import matplotlib.pyplot as plt
import PIL
from functions import *

import numpy as np

# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

import functions
# Import image
images = []
for i in range(1, 65):
    num = str(i) if i >= 10 else "0" + str(i)
    image = PIL.Image.open(f"./toyProblem_F22/frame_{num}.png").convert("L")
    image = np.asarray(image)
    # image = np.reshape(image, (-1,))
    # images.append(np.asarray(image, dtype=np.float32)/255)
    images.append(image)
    
# Calculate optic flow for all frames
interval = 5      # How many optical flow vectors to calculate
n = 3               # Size of neighboorhood in Lucas-Kanade method
opticFlowX, opticFlowY = functions.optical_flow(np.asarray(images), interval, n)

frameNr = 3
for i in range(63):
    plotVectorField(np.asarray(images[i]), opticFlowX[i], opticFlowY[i], i)

# print(functions.plotVectorField(images[i],opticFlowX[i],opticFlowY[i],i).shape)
# for i in range(64):
#     plotVectorField(images[i],opticFlowX[i],opticFlowY[i],i)

# MOVIE STUFF 
# numpy array
# x = np.linspace(-2, 2, 200)


# duration of the video and the FPS
duration = 5
FPS = 10

# matplot subplot
fig, ax = plt.subplots()
 
# method to get frames
def make_frame(time):
    # clear
    ax.clear()
     
    frame = int(time*FPS)

    # plotting line
    #ax.imshow(images[frame], cmap="gray")
     
    # plotting just vector field
    idx_x = np.arange(256)
    idx_y = np.arange(256)
    idx_x,idx_y = np.meshgrid(idx_x, idx_y)
    
    #Ignore all nonzero entries
    mask = np.logical_or(opticFlowX != 0,opticFlowY !=0)
    
    X = idx_x[mask]
    Y = idx_y[mask]
    U = opticFlowX[mask]
    V = opticFlowY[mask]

    ax.quiver(X,Y,U,V, scale = 100)

    img = ax.get_figure()

    # returning numpy image
    return np.asarray(img)

# creating animation
animation = VideoClip(make_frame, duration = duration)

animation.write_gif("./gifs/test_flow2.gif",fps=FPS)