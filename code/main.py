import matplotlib.pyplot as plt
import numpy as np
import PIL

from video_plotter import save_video
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
interval = 5        # How many optical flow vectors to calculate
n = 3               # Size of neighboorhood in Lucas-Kanade method
opticFlowX, opticFlowY = functions.optical_flow(np.asarray(images), interval, n)

frameNr = 3
for i in range(63):
    functions.plotVectorField(np.asarray(images[i]), opticFlowX[i], opticFlowY[i], i)


save_video(image_folder_from="toyProblem_F22_vectorField", N_IMAGES=63, video_name_to="test")
