import matplotlib.pyplot as plt
import numpy as np
import PIL

# Save images to array
images = []
for i in range(1, 65):
    num = str(i) if i >= 10 else "0" + str(i)
    image = PIL.Image.open(f"./toyProblem_F22/frame_{num}.png").convert("L")
    images.append(np.asarray(image, dtype=np.float32)/255)

# Plot images in sequence
# fig, ax = plt.subplots()
# for i, img in enumerate(images):
#     ax.clear()
#     ax.imshow(img, cmap="gray")
#     ax.set_title(f"frame {i}")
#     # Note that using time.sleep does *not* work here!
#     plt.pause(0.05)

# with open("image_3d.txt", "w") as f:
#     f.write("")
#     f.close()

# with open("image_3d.txt", "w") as f:
#     f.write(str([list(]) for l in im]))


# Reshape images to one long(!) vector
images_vector = np.array(images, dtype=np.float32)
flattened_images_vector = np.reshape(images_vector, (-1,))
print(flattened_images_vector.shape)

depth, height, width = images_vector.shape

length = len(flattened_images_vector)

def calculate_Vx(flattened_image_vector, length, width, height):
    Vx = flattened_image_vector[1:] - flattened_image_vector[:-1]
    Vx = np.append(Vx,0)
    for i in range(length):
        if (i+1) % width == 0:
            Vx[i] = 0
    return Vx

def calculate_Vy(flattened_image_vector, length, width, height):
    Vy = flattened_image_vector[width:] - flattened_image_vector[:-width]
    Vy = np.append(Vy, np.zeros(width))
    for i in range(length):
        if i % (width * height) >= width * (height - 1):
            Vy[i] = 0
    return Vy

def calculate_Vt(flattened_image_vector, length, width, height):
    Vt = flattened_image_vector[width * height:] - flattened_image_vector[:-width * height]
    Vt = np.append(Vt, np.zeros(width*height))
    return Vt


##############
# TEST CASES #
##############

a = np.array([
    [
        [-1, -2, -3], 
        [-4, -5, -6]
    ], 
    [
        [7, 8, 9],
        [10, 11, 12]
    ],
    [
        [13, 14, 15],
        [16, 17, 18]
    ],
    [
        [19, 20, 21],
        [22, 23, 24]
    ]
    ])
b = a.reshape((-1, ))
print("a")
print(a)

depth, height, width = a.shape
length = len(b)

Vx = calculate_Vx(b, length, width, height)
Vx = np.reshape(Vx, (depth, height, width))
print("Vx")
print(Vx)

Vy = calculate_Vy(b, length, width, height)
Vy = np.reshape(Vy, (depth, height, width))
print("Vy")
print(Vy)

Vt = calculate_Vt(b, length, width, height)
Vt = np.reshape(Vt, (depth, height, width))
print("Vt")
print(Vt)