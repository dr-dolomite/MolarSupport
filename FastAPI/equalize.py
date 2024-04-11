import matplotlib.pyplot as plt
import numpy as np
import cv2

nemo = cv2.imread('image.png')
nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)


light_green = (73, 143, 92)
dark_green = (25, 93, 38)

mask = cv2.inRange(hsv_nemo, light_green, dark_green)

result = cv2.bitwise_and(nemo, nemo, mask=mask)

from matplotlib.colors import hsv_to_rgb

lo_square = np.full((10, 10, 3), light_green, dtype=np.uint8) / 255.0
do_square = np.full((10, 10, 3), dark_green, dtype=np.uint8) / 255.0

plt.subplot(1, 2, 1)
plt.imshow(hsv_to_rgb(do_square))
plt.subplot(1, 2, 2)
plt.imshow(hsv_to_rgb(lo_square))
plt.show()


