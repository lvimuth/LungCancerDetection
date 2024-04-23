import numpy as np
import cv2
import math
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
from scipy import ndimage, misc
#import FFT as f

#img_title= f.img_title
#img_title= "C:/Users/Dom/Downloads/1.2.826.0.1.3680043.2.656.1.138.181.jpg"
img_title = 'cancer.jpg'
medianfiltersize = 10

def masking(img_title):
    image = cv2.imread(img_title, cv2.COLOR_BGR2GRAY)
    image= cv2.resize(image,(512,512))
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    figure_size=9
    median_mask = cv2.medianBlur(image2, figure_size)
    #median_mask=ndimage.median_filter(image2, size=medianfiltersize)
    image_sharp = cv2.addWeighted(image, 2, image, -1, 0)
    
    initialContoursImage = np.copy(median_mask)
    imgray = cv2.cvtColor(initialContoursImage, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 127, 255, 0)
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

    biggestContourImage = np.copy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.drawContours(biggestContourImage, [cnt], -1, (0,0,255), 3)

    filled = np.zeros_like(thresh)
    x,y,w,h = cv2.boundingRect(cnt)
    mask=cv2.drawContours(filled, [cnt], 0, 255, -1)
    plt.imshow(mask)
    plt.show()
    width_half = image.shape[1]//2
    x_half_mask = filled.shape[0]//2
    img_to_mask = image[:,width_half-x_half_mask:width_half+x_half_mask]
    masked = cv2.bitwise_and(img_to_mask,img_to_mask,mask = filled)
    pixels = cv2.countNonZero(mask)
    return masked,pixels

img,pixels =masking(img_title)
print(pixels)

                # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
                # ksize - size of gabor filter (n, n)
                # sigma - standard deviation of the gaussian function
                # theta - orientation of the normal to the parallel stripes
                # lambda - wavelength of the sunusoidal factor
                # gamma - spatial aspect ratio
                # psi - phase offset
                # ktype - type and range of values that each pixel in the gabor kernel can hold

theta = (np.pi)*6
g_kernel = cv2.getGaborKernel((10, 10), 3, theta/16, 5, 0.75, 0, ktype=cv2.CV_32F)

                #img = masked_img#cv2.imread('C:/Users/Dom/Downloads/1.2.826.0.1.3680043.2.656.1.138.181.jpg',)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

plt.subplot(121)
plt.title('Kernal')
plt.imshow(g_kernel,cmap='gray')
                #plt.show()
h, w = g_kernel.shape[:2]
g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
plt.subplot(122)
plt.title('Filtered Image')
plt.imshow(filtered_img,cmap='gray')

plt.show()

