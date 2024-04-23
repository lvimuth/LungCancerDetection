import numpy as np
import cv2
from scipy import ndimage, misc
import math
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
import Otsu_Thresh as ot
import numpy as np
from PIL import Image

#img_title= 'test.png'
#image = cv2.imread(img_title, cv2.COLOR_BGR2GRAY)
image = ot.mask
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


_, thresh = cv2.threshold(image, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
largest_area=0
index = 0
for contour in contours:
    if index > 0:
        area = cv2.contourArea(contour)
        if (area>largest_area):
            largest_area=area
            cnt = contours[index]
    index = index + 1

image = cv2.resize(image,(512,512))
cv2.drawContours(image, [cnt], -1, (0,0,255), 3)

src1 = ot.path
src2 = image#mask#image

#img = np.array(img, dtype=np.uint8)

#src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#dst = src * image

#Image.fromarray(dst.astype(np.uint8)).save('numpy_image_mask.jpg')

dst = cv2.bitwise_and(src1, src2)

#cv2.imwrite('opencv_bitwise_and.jpg', dst)
plt.subplot(121)
plt.imshow(src1,cmap='gray')
plt.title("Input Image")
plt.subplot(122)
plt.imshow(dst,cmap='gray')
plt.title("Nodule detected Image")
plt.savefig('Nodule detected Image.png')
plt.show()
