import numpy as np
import scipy

from functions import calculate_Vx_SGF, calculate_Vy_SGF, calculate_Vt_SGF

images = np.array([
    [[1,1,1],[1,2,3],[9,9,8]],
    [[1,1,1],[1,2,3],[9,9,8]],
    [[1,1,1],[1,2,3],[9,9,8]],
])
print(images.shape)
print(images[0])


print(calculate_Vx_SGF(images))


