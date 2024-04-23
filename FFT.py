import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,exp
import FFT_filters as f
import CancerDetection as cd

img_title= cd.img_title
#img_title= "C:/Users/Dom/Desktop/Sem07/cancer1.png"
#"C:/Users/Dom/Desktop/Sem07/Normal.png"
#"C:/Users/Dom/Desktop/Sem07/cancer2.jpg"
img = cv2.imread(img_title,0)
img= cv2.resize(img,(512,512))
original = np.fft.fft2(img)
center = np.fft.fftshift(original)

#plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)

#plt.subplot(151), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Spectrum")

LowPass = f.idealFilterLP(50,img.shape)
#plt.subplot(152), plt.imshow(np.abs(LowPass), "gray"), plt.title("Low Pass Filter")

LowPassCenter = center * f.idealFilterLP(50,img.shape)
#plt.subplot(153), plt.imshow(np.log(1+np.abs(LowPassCenter)), "gray"), plt.title("Centered Spectrum multiply Low Pass Filter")

LowPass = np.fft.ifftshift(LowPassCenter)
#plt.subplot(154), plt.imshow(np.log(1+np.abs(LowPass)), "gray"), plt.title("Decentralize")

inverse_LowPass = np.fft.ifft2(LowPass)
#plt.subplot(155), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Processed Image")

fft_image = np.abs(inverse_LowPass)
plt.imshow(fft_image, "gray"), plt.title("Processed Image with FT")

plt.show()
