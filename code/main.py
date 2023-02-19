import matplotlib.pyplot as plt
import numpy as np
import PIL

from video_plotter import save_video
import functions

# Import image
images = []
for i in range(80):
    num = str(i) if i >= 10 else "0" + str(i) #use for toyProblem
    image = PIL.Image.open(f"./rulleboldImages/frame{str(i)}.png").convert("L")
    image = np.asarray(image)/255
    images.append(image)
    
# Calculate optic flow for all frames
interval = 43        # How many optical flow vectors to calculate
n = 128               # Size of neighboorhood in Lucas-Kanade method
opticFlowX, opticFlowY = functions.optical_flow(np.asarray(images), interval, n)

for i in range(70):
    functions.plotVectorField(np.asarray(images[i]), opticFlowX[i], opticFlowY[i], i)


save_video(image_folder_from="rulleboldImages_vectorField", N_IMAGES=65, video_name_to="rullebold")
