import os

"""
Quickly rename vante images
"""

r = os.listdir("./vanteImages")

for i in r:
    part1, part2 = i.split("e")
    num, format = part2.split(".")
    num = int(num)
    # print((num, num + 1))
    os.rename("./vanteImages/" + str(i), "./vanteImages/frame_" + str(num + 1) + ".png")