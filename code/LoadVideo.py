import cv2


vidcap = cv2.VideoCapture('/Users/tuxen/Desktop/02526 Mathematical Modeling/optic flow/02526-mathematical-modeling-project-1/code/data/rullebold.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("/Users/tuxen/Desktop/02526 Mathematical Modeling/optic flow/02526-mathematical-modeling-project-1/code/rulleboldImages/frame%d.png" % count, image)     # save frame as PNG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1