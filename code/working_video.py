import matplotlib.pyplot as plt
import PIL

import numpy as np

# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


# Import image
N_IMAGES = 64

images = []
for i in range(1, N_IMAGES+1):
    num = str(i) if i >= 10 else "0" + str(i)
    image = PIL.Image.open(f"./toyProblem_F22/frame_{num}.png").convert("L")
    image = np.asarray(image)
    # image = np.reshape(image, (-1,))
    # images.append(np.asarray(image, dtype=np.float32)/255)
    images.append(image)

# duration of the video and the FPS
FPS = 10
duration = N_IMAGES/FPS

# matplot subplot
fig, ax = plt.subplots()
 
# method to get frames
def make_frame(time):
    # clear
    ax.clear()
    frame = int(time*FPS)

    # plotting image
    ax.imshow(images[frame], cmap="gray")
     
    # returning numpy image
    return images[frame]

# creating animation
animation = VideoClip(make_frame, duration = duration)

animation.write_gif("./gifs/working_video_functionality.gif",fps=FPS)