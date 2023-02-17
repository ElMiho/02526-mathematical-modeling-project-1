import matplotlib.pyplot as plt
import numpy as np
import PIL

from video_plotter import save_video
import functions

# Import image
images = []
for i in range(40, 110):
    num = str(i) if i >= 10 else "0" + str(i) #use for toyProblem
    image = PIL.Image.open(f"./vanteImages/frame{str(i)}.png").convert("L")
    image = np.asarray(image)/255
    images.append(image)
    
# Calculate optic flow for all frames
interval = 5        # How many optical flow vectors to calculate
n = 3               # Size of neighboorhood in Lucas-Kanade method
opticFlowX, opticFlowY = functions.optical_flow(np.asarray(images), interval, n)

for i in range(70):
    functions.plotVectorField(np.asarray(images[i]), opticFlowX[i], opticFlowY[i], i)


save_video(image_folder_from="vanteImages_vectorField", N_IMAGES=70, video_name_to="vantegif")
