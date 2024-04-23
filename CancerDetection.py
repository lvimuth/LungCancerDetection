import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from keras import models


img_title="cancer.jpg"
#img_title= "C:/Users/Dom/Desktop/Sem07/cancer1.png"
#"C:/Users/Dom/Desktop/Sem07/Normal.png"
#"C:/Users/Dom/Desktop/Sem07/cancer.jpg"
model = load_model('03model.h5')

image = cv2.imread(img_title,flags=cv2.IMREAD_COLOR)        
#image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)     
image = cv2.resize(image,(250,250))
img=image

image = np.expand_dims(image, axis=0)
image = np.array(image)/255
test_image =image

def ClassPredict(argument):
    #1:Adenocarcinoma
    #2:Large Cell Carcinoma
    #3:Squamous Cell CArcinoma
    switcher = {
        0: " Normal, No detected cancer",
        1: " Patient has a lung cancer  ",
        2: " Patient has a lung cancer  ",
        3: " Patient has a lung cancer  "
    }
    return switcher.get(argument, "nothing")



plt.imshow(img)
plt.title("Input Image")
plt.show()
    
result = ClassPredict(model.predict_classes(test_image)[0])
plt.imshow(img)
print(result)
plt.title(result)
plt.show()

import Intermediate
