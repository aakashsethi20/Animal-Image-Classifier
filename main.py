import numpy as np
import matplotlib.pyplot as plt
import os, sys
import caffe

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# checking if the model exists, else it will exit the program
# and ask the user to put the caffemodel at the appropriate location
if os.path.isfile('./models/bvlc_reference_caffenet.caffemodel'):
    print('Caffemodel found!')
else:
    print('Caffemodel not at the correct place. Please download it and place it at ./models')
    sys.exit(1)

# setting the caffe to use CPU since my computer doesn't have
# a CUDA enabled GPU
caffe.set_mode_cpu()

