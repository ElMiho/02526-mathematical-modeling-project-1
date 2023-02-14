import matplotlib.pyplot as plt
import PIL

import numpy as np
import os

# importing movie py libraries
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

def save_video(image_folder_from: str, N_IMAGES: int, video_name_to: str, FPS:int=5, duration_factor:float=2.5):
    # Import images
    images = []
    image_folder_ims = sorted([(int(im[:-4]), im) for im in os.listdir(image_folder_from)])
    for image_name in [image_name for _, image_name in image_folder_ims]:
        image = PIL.Image.open(f"./{image_folder_from}/{image_name}").convert("L")
        image = np.asarray(image)
        # image = np.reshape(image, (-1,))
        # images.append(np.asarray(image, dtype=np.float32)/255)
        images.append(image)

    # duration of the video and the FPS
    
    duration = duration_factor * N_IMAGES/FPS

    # matplot subplot
    fig, ax = plt.subplots()
    
    # method to get frames
    def make_frame(time):
        # clear
        ax.clear()
        frame = int(time*FPS/duration_factor)
        print(frame)
        # plotting image
        ax.imshow(images[frame], cmap="gray")
        
        # returning numpy image
        return images[frame]

    # creating animation
    animation = VideoClip(make_frame, duration = duration)

    animation.write_gif(f"./gifs/{video_name_to}.gif",fps=FPS)