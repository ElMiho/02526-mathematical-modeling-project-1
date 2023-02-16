import cv2


vidcap = cv2.VideoCapture('./vante.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("vanteImages/frame%d.png" % count, image)     # save frame as PNG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1