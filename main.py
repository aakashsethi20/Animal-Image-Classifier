import numpy as np
import matplotlib.pyplot as plt
import os, sys, subprocess
import caffe

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# Setting the path for the models
models_path = './models/'

# checking if the model exists, else it will exit the program
# and ask the user to put the caffemodel at the appropriate location
if os.path.isfile(models_path + 'bvlc_reference_caffenet.caffemodel'):
    print('Caffemodel found!')
else:
    print('Caffemodel not at the correct place. Please download it and place it at ./models')
    sys.exit(1)

# setting the caffe to use CPU since my computer doesn't have
# a CUDA enabled GPU
caffe.set_mode_cpu()

# Setting the path to model definition and trained model weights
model_def = models_path + 'deploy.prototxt'                         # Deliverable B
model_weights = models_path + 'bvlc_reference_caffenet.caffemodel'  # Deliverable C

# Initializing caffe's Net class with the pretrained model and definition
net = caffe.Net(model_def, model_weights, caffe.TEST) # caffe.NET sets the application in Test mode

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('./imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print ('mean-subtracted values:', list(zip('BGR', mu)))

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))      # move image channels to outermost dimension
transformer.set_mean('data', mu)                # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)          # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))   # swap channels from RGB to BGR

# set the size of the input
net.blobs['data'].reshape(7,         # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 640x668

image = caffe.io.load_image('./images/dog.jpeg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print ('predicted class is:', output_prob.argmax())

# load ImageNet labels
labels_file = './data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    subprocess.call(['./data/ilsvrc12/get_ilsvrc_aux.sh'])
    
labels = np.loadtxt(labels_file, str, delimiter='\t')

print ('output label:', labels[output_prob.argmax()])

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print ('probabilities and labels:')
print (list(zip(output_prob[top_inds], labels[top_inds])))