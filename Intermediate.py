from keras import models
import CancerDetection as cd
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tensorflow as tf

model = cd.model
test_image = cd.test_image
layer_outputs = [layer.output for layer in model.layers[:6]] 
# Extracts the outputs of the top 12 layers
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activations = activation_model.predict(test_image) 
# Returns a list of five Numpy arrays: one array per layer activation
first_layer_activation = activations[0]
#print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='gray')
plt.title("Input Layer")
plt.show()
layer_names = []
for layer in model.layers[:6]:
  layer_names.append(layer.name) # Names of the layers
    
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
  n_features = layer_activation.shape[-1] # Number of features in the feature map
  size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
  n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
  display_grid = np.zeros((size * n_cols, images_per_row * size))
  for col in range(n_cols): # Tiles each filter into a big horizontal grid
    for row in range(images_per_row):
      channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
      channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
      channel_image /= channel_image.std()
      np.seterr(divide='ignore', invalid='ignore')
      channel_image *= 64
      channel_image += 128
      channel_image = np.clip(channel_image, 0, 255).astype('uint8')
      display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
  scale = 1. / size
  fig=plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
  plt.title(layer_name)
  plt.grid(False)
  plt.imshow(display_grid, aspect='auto', cmap='gray')
  fig.savefig("featuremap-layer-{}".format(layer_name)+'.jpg')
  plt.show()
